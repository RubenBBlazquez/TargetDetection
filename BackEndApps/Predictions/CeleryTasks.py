import json

from celery import app
import requests
from Core.Celery.Celery import app as celeryApp
from datetime import datetime
from BackEndApps.Predictions.models import RawPredictionData, GoodPredictions, CleanPredictionData, AllPredictions
import logging
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection
import pickle
import os
import numpy as np
import pandas as pd
from BackEndApps.Predictions.services.DistanceCalculations import DistanceCalculations
from RaspberriModules.DataClasses.ServoModule import ServoMovement
from RaspberriModules.DataClasses.PowerModule import PowerModule
import time
import RPi.GPIO as GPIO

@app.shared_task
def purge_celery(*args):
    """
    This task is used to clean all tasks in celery queue

    Parameters:
    -----------
    *args: tuple
        This tuple contains the following arguments:
            - queue_name: str
                This argument contains the name of the queue to purge.
    """
    queue_name = args[0]
    api_url = f'{os.getenv("RABBITMQ_URL_API")}/queues/{os.getenv("RABBITMQ_VHOST")}/{queue_name}'
    response = requests.get(api_url)

    if response.status_code != 200:
        return

    queue_info = response.json()
    ready_tasks = int(queue_info['messages_ready'])

    if ready_tasks > 15:
        print('purging unnecesary tasks')
        celeryApp.control.purge()


@app.shared_task
def check_prediction(*args):
    """
        This task is used to check if an image obtained from raspberri is a correct prediction
        using the YoloTargetDetection Model.
        If prediction is correct, we launch a task to start the prediction actions.

        Parameters:
        -----------
        *args: tuple
            This tuple contains the following arguments:
                - image_bytes: bytes
                    This argument contains the image in bytes format.
                - servo_position: int
                    This argument contains the servo position when the image was taken.
                - date: datetime
                    This argument contains the date when the image was taken.
    """
    image_bytes, servo_position, date = args
    image = pickle.loads(image_bytes)

    logging.info('----------------------------------------------------------------------')
    logging.info(f'TASK (check_prediction) RECEIVED: {datetime.now()}')

    if datetime.now().day != date.day:
        logging.info(
            f'TASK (check_prediction) REJECTED: DATE DAY ({date.day}) ARGUMENT IS DIFFERENT FROM ACTUAL DATE ({datetime.now().day})')

        return

    model = YoloTargetDetection(os.getenv('YOLO_MODEL_NAME'))
    result_prediction = model.predict(image)
    labels = result_prediction.pandas().xywh[0]
    predicted_labels = labels[labels['confidence'] > 0.60]
    logging.info(f'Predicted Labels: \n {predicted_labels}')

    prediction_object = AllPredictions(
        image=json.dumps(image.tolist()),
        prediction=int(len(predicted_labels) > 0),
        confidence=float(predicted_labels['confidence'].mean())
    )
    prediction_object.save()
    prediction_id = str(prediction_object._id)

    if not predicted_labels.empty:
        predicted_labels.apply(
            lambda p_labels:
            start_predictions_ok_actions.apply_async(
                RawPredictionData(
                    pickle.dumps(image),
                    pickle.dumps(p_labels),
                    str(prediction_id),
                    servo_position,
                    datetime.now()
                ),
                ignore_result=True,
                queue='prediction_ok_actions',
                priority=10
            ),
            axis=1
        )


@app.shared_task
def start_predictions_ok_actions(*args):
    """
        This task is used to launch a task to manage the prediction and save it on database.

        Parameters:
        -----------
        *args: tuple
            This tuple contains the following arguments:
                - image_bytes: bytes
                    This argument contains the image in bytes format.
                - labels_bytes: bytes
                    This argument contains the labels from the prediction in bytes format.
                - prediction_id: str
                    This argument contains the prediction id.
                - servo_position: int
                    This argument contains the servo position when the image was taken.
                - date: datetime
                    This argument contains the date when the image was taken.
    """
    image_bytes, labels_bytes, prediction_id, servo_position, date = args
    original_image: np.ndarray = pickle.loads(image_bytes)
    labels: pd.Series = pickle.loads(labels_bytes)

    if datetime.now().day != date.day:
        logging.info(
            f'TASK (launch_prediction_action) REJECTED: DATE DAY ({date.day}) ARGUMENT '
            f'IS DIFFERENT FROM ACTUAL DATE ({datetime.now().day})')

        return

    image = original_image.copy()
    distance_calculations = DistanceCalculations.create_from(image, labels)
    distance_calculations.draw_lines_into_image()

    GoodPredictions.create_from(
        CleanPredictionData(
            json.dumps(labels.to_dict()),
            json.dumps(image.tolist()),
            json.dumps(distance_calculations.get_all_distances().to_dict()),
            servo_position,
            prediction_id
        )
    ).save()

    celeryApp.control.purge()
    is_shoot_in_progress = os.path.exists('RaspberriModules/assets/shoot_in_progress.tmp')

    if is_shoot_in_progress:
        return

    # we move the servo to the position where the target is
    x_servo = ServoMovement(int(os.getenv('X_SERVO_PIN')), servo_position)
    x_servo.default_move()
    time.sleep(0.5)
    calculate_shoot_position(distance_calculations.get_all_distances(), x_servo)

def calculate_shoot_position(calculated_distances: pd.Series, servo_x: ServoMovement):
    """
        This task is used to calculate the shot position (move servo to the correct position to shoot the target).

        Parameters:
        -----------
        - calculated_distances: pd.Series
            This argument contains the distances to all sides of the target
    """
    tmp_file = f'RaspberriModules/assets/shoot_in_progress.tmp'
    # we create a tmp file to indicate that we are calculating the shoot position
    # and the real time prediction must be stopped
    open(tmp_file, 'w').close()
    # we calculate the shoot position
    left = calculated_distances.left
    right = calculated_distances.right
    top = calculated_distances.top
    bottom = calculated_distances.bottom

    center = (left + right) / 2
    center_top = (top + bottom) / 2

    shoot_position = (center, center_top)

    y_servo = ServoMovement(int(os.getenv('Y_SERVO_PIN')), 11)
    laser = PowerModule(int(os.getenv('LASER_PIN')))

    laser.on()
    y_servo.default_move()
    y_servo.stop()
    time.sleep(1)
    laser.off()

    os.remove(tmp_file)
