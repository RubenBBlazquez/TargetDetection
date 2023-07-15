import json

from celery import app
import requests
from Core.Celery.Celery import app as celeryApp
from datetime import datetime
from Predictions.models import RawPredictionData, Predictions, CleanPredictionData
import logging
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection
import pickle
import os
import numpy as np
import pandas as pd
from Predictions.services.DistanceCalculations import DistanceCalculations


@app.shared_task
def purge_celery():
    """
    This task is used to clean all tasks in celery queue
    """
    api_url = f'{os.getenv("RABBITMQ_URL_API")}/queues/{os.getenv("RABBITMQ_VHOST")}/YoloPredictions'
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
        This task is used to check if prediction is correct, when we receive a prediction from YoloTargetDetection.
        If prediction is correct, we launch a task to manage the prediction and save on database.

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
    predicted_labels = result_prediction.pandas().xywh[0]

    logging.info(predicted_labels['confidence'] > 0.60)

    predicted_labels = predicted_labels[predicted_labels['confidence'] > 0.60]

    logging.info(f'Predicted Labels: \n {predicted_labels}')

    if not predicted_labels.empty:
        predicted_labels.apply(
            lambda labels:
            launch_prediction_action.apply_async(
                RawPredictionData(pickle.dumps(image), pickle.dumps(labels), servo_position, datetime.now()),
                ignore_result=True,
                queue='YoloPredictions',
                priority=10
            ),
            axis=1
        )


@app.shared_task
def launch_prediction_action(*args):
    """
        This task is used to launch a task to manage the prediction and save on database.

        Parameters:
        -----------
        *args: tuple
            This tuple contains the following arguments:
                - image_bytes: bytes
                    This argument contains the image in bytes format.
                - labels_bytes: bytes
                    This argument contains the labels from the prediction in bytes format.
                - servo_position: int
                    This argument contains the servo position when the image was taken.
                - date: datetime
                    This argument contains the date when the image was taken.
    """
    image_bytes, labels_bytes, servo_position, date = args
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

    Predictions.create_from(
        CleanPredictionData(
            json.dumps(original_image.tolist()),
            json.dumps(labels.to_dict()),
            json.dumps(image.tolist()),
            json.dumps(distance_calculations.get_all_distances()),
            servo_position
        )
    ).save()

    celeryApp.control.purge()


@app.shared_task
def calculate_shoot_position(*args):
    """
        This task is used to calculate the shot position (move servo to the correct position to shoot the target).

        Parameters:
        -----------
        *args: tuple
            This tuple contains the following arguments:
                - distances: bytes
                    This argument contains the distances to all sides of the target
    """
    distances = args[0]
    distances = pickle.loads(distances)
    tmp_file = 'RasperriModules/assets/shoot_in_progress.tmp'
    # we create a tmp file to indicate that we are calculating the shoot position
    # and the real time prediction must be stopped
    open(tmp_file, 'w').close()

    # we calculate the shoot position
    shoot_position = distances.get_shoot_position()

    os.remove(tmp_file)
