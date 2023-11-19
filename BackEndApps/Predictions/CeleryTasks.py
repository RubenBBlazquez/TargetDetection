import json
from celery import app
import requests
from Core.Celery.Celery import purge_specific_queue
from datetime import datetime
from BackEndApps.Predictions.models import RawPredictionData, GoodPredictions, CleanPredictionData, AllPredictions
import logging
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection
import pickle
import os
import numpy as np
import pandas as pd
from BackEndApps.Predictions.services.DistanceCalculations import DistanceCalculations
from RaspberriModules.DataClasses.ServoModule import ServoMovement, ServoManagement
from RaspberriModules.DataClasses.PowerModule import PowerModule
import time

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

    if ready_tasks >= 20:
        print(f'purging unnecesary tasks from {queue_name}')
        purge_specific_queue(queue_name)
        purge_specific_queue("purge_data")


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

    is_shoot_in_progress = os.path.exists('RaspberriModules/assets/shoot_in_progress.tmp')
    if is_shoot_in_progress:
        return

    ServoManagement().servos = {}
    x_servo = ServoMovement(int(os.getenv('X_SERVO_PIN')), servo_position, name="x2")
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

    duty_cycle_per_cm = float(os.getenv('DUTY_CYCLE_PER_CM', 0.5))
    servo_duty_cycle_position = servo_x.position
    y_servo = ServoMovement(int(os.getenv('Y_SERVO_PIN', 0)), 0, name='y1')
    y_servo_medium_position = float(os.getenv('Y_SERVO_MEDIUM_POSITION_CYCLE', 0))
    y_servo_bottom_max_position = float(os.getenv('Y_SERVO_BOTTOM_POSITION_CYCLE', 0))
    y_servo_top_max_position = float(os.getenv('Y_SERVO_MAX_TOP_POSITION_CYCLE', 0))

    # we calculate the shoot position
    left = calculated_distances.left
    right = calculated_distances.right
    top = calculated_distances.top
    bottom = calculated_distances.bottom
    from_top_side_to_center = calculated_distances.from_top_side_to_center
    from_bottom_side_to_center = calculated_distances.from_bottom_side_to_center

    target_width = calculated_distances.width - (right + left)
    center_target_x = target_width/2
    center_x_where_image_was_taken = (calculated_distances.width/2) -  int(os.getenv("DISTANCE_CAMERA_ERROR_CM", 3))

    # we are using this formula to calculate if the target is centered:
    # (center_image - 0.5 <= (right + center_target_x) <= center_image + 0.5)
    # if we have the image (witdh = 100) and (left = 44.50) and (right = 45) and the target has 10 cm size
    # so the formula to know if is centered is: 50 - 0.5 <= (45 + 5) <= 50 + 0.5 = true
    is_target_centered = center_x_where_image_was_taken - 0.5 <= (right + center_target_x) <= center_x_where_image_was_taken + 0.5

    if is_target_centered:
        y_servo.move_to(y_servo_medium_position)
        time.sleep(1)
        y_servo.stop()

        shoot()

    if not is_target_centered:
        if right > left:
            cm_to_center = left + center_target_x
            duty_cycle_position = servo_duty_cycle_position + (cm_to_center * duty_cycle_per_cm)
            duty_cycle_position = duty_cycle_position if duty_cycle_position > 0 else 1
            print(f"1111, {servo_duty_cycle_position} {left} {center_target_x} {duty_cycle_position}")
            servo_x.move_to(duty_cycle_position)
        else:
            cm_to_center = right + center_target_x
            duty_cycle_position = servo_duty_cycle_position - (cm_to_center * duty_cycle_per_cm)
            duty_cycle_position = duty_cycle_position if duty_cycle_position > 0 else 0
            print(f"2222, {servo_duty_cycle_position} {right} {center_target_x} {duty_cycle_position}")
            servo_x.move_to(duty_cycle_position)

        time.sleep(1)
        servo_x.stop()

    if bottom < top:
        duty_cycle_position = y_servo_medium_position + (from_top_side_to_center * duty_cycle_per_cm)
        y_servo.move_to(
            duty_cycle_position if duty_cycle_position > y_servo_bottom_max_position else y_servo_bottom_max_position
        )
    else:
        duty_cycle_position = y_servo_medium_position - (from_bottom_side_to_center * duty_cycle_per_cm)
        y_servo.move_to(
            duty_cycle_position if duty_cycle_position < y_servo_top_max_position else y_servo_top_max_position
        )

    time.sleep(1)
    y_servo.stop()
    shoot()

    y_servo = ServoMovement(int(os.getenv('Y_SERVO_PIN', 0)), 0, name='y1')
    y_servo.move_to(y_servo_medium_position)
    time.sleep(1)
    y_servo.stop()

    purge_specific_queue("prediction_ok_actions")
    os.remove(tmp_file)

def shoot():
    laser = PowerModule(int(os.getenv('LASER_PIN', 0)))
    buzzer = PowerModule(int(os.getenv('BUZZER_PIN', 0)))

    laser_speed = 0.55
    for x in range(1, 50):
        laser.on()
        buzzer.on()
        time.sleep(laser_speed)

        laser.off()
        buzzer.off()
        time.sleep(laser_speed)

        if x % 3 == 0 and laser_speed > 0.10:
            laser_speed = laser_speed - 0.10
