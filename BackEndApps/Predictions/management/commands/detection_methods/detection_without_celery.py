import json
import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from BackEndApps.Predictions.ml_models.real_turret_env import get_image_from_camera, CameraType
from BackEndApps.Predictions.models import RawData, AllPredictions, GoodPredictions, CleanPredictionData
from BackEndApps.Predictions.services.DistanceCalculations import DistanceCalculations
from Core.Services.TargetDetection.YoloV8TargetDetection import YoloV8TargetDetection


def real_time_detection():
    """
       This function is used to start the real time detection using celery tasks to predict targets and not predict in the
       same thread as the camera.
       """
    # we import here to avoid errors when we are not execution on the raspberry pi
    from RaspberriModules.DataClasses.CustomPicamera import CustomPicamera
    from RaspberriModules.DataClasses.ServoModule import ServoMovement

    picamera = CustomPicamera()
    picamera.start()
    frames = 0
    angle = 1
    gpin_horizontal_servo = int(os.getenv('X_SERVO_PIN'))
    increment = 1
    servo_movements = 0
    cv2.startWindowThread()
    servo = ServoMovement(gpin_horizontal_servo, angle, name='x1')

    while True:
        frames += 1
        image = get_image_from_camera(picamera, CameraType.CSI)
        cv2.imshow("CSI Camera", image)
        cv2.waitKey(1)
        is_shoot_in_progress = os.path.exists('RaspberriModules/assets/shoot_in_progress.tmp')

        if frames < 15:
            continue

        if is_shoot_in_progress:
            servo.stop()
            continue

        if angle < 0:
            angle = 1

        servo = ServoMovement(gpin_horizontal_servo, angle, name='x1')
        servo.move_to(angle)
        time.sleep(0.4)

        servo_movements += 1
        check_prediction(image, angle)

        if servo_movements % 10 == 0:
            increment = -increment

        angle += increment
        frames = 0

    cv2.destroyAllWindows()


def check_prediction(image: np.array, servo_position):
    """
    This function is used to check if the prediction is ok or not and save the result in the database.

    Parameters
    ----------
    image: np.array
        This is the image that we want to predict.
    servo_position: int
        This is the servo position when the image was taken.
    date: datetime
        This is the date when the image was taken.
    """
    model = YoloV8TargetDetection(os.getenv('YOLO_MODEL_NAME'))
    predicted_labels = model.predict_and_filter_by_confidence(image, 0.1)

    prediction_object = AllPredictions(
        image=json.dumps(image.tolist()),
        prediction=int(len(predicted_labels) > 0),
        confidence=float(predicted_labels['confidence'].mean())
    )
    prediction_object.save()
    prediction_id = str(prediction_object._id)

    for index, labels in predicted_labels.iterrows():
        start_predictions_ok_actions(
            image,
            labels,
            prediction_id,
            servo_position,
        )


def start_predictions_ok_actions(image, labels, prediction_id, servo_position):
    """
        This task is used to launch a task to manage the prediction and save it on database.

        Parameters:
        -----------
        image: np.array
            This is the image that we want to predict.
        labels: pd.Series
            This is the labels of the image.
        prediction_id: str
            This is the id of the prediction.
        servo_position: int
            This is the servo position when the image was taken.
        date: datetime
            This is the date when the image was taken.
    """
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

    # we import here to avoid errors when we are not execution on the raspberry pi
    from RaspberriModules.DataClasses.ServoModule import ServoManagement, ServoMovement

    ServoManagement().servos = {}
    x_servo = ServoMovement(int(os.getenv('X_SERVO_PIN')), servo_position, name="x2")
    calculate_shoot_position(distance_calculations.get_all_distances(), x_servo)


def calculate_shoot_position(calculated_distances: pd.Series, servo_x: "ServoMovement"):
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

    # we import here to avoid errors when we are not execution on the raspberry pi
    from RaspberriModules.DataClasses.ServoModule import ServoMovement

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
    from_x_center_to_center_image = calculated_distances.from_x_center_to_center_image

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
        duty_cycle_position = servo_duty_cycle_position + (from_x_center_to_center_image * duty_cycle_per_cm)

        if right < left:
            duty_cycle_position = servo_duty_cycle_position - (from_x_center_to_center_image * duty_cycle_per_cm)

        duty_cycle_position = duty_cycle_position if duty_cycle_position > 2 else 2
        duty_cycle_position = duty_cycle_position if duty_cycle_position < 12 else 12
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

def shoot():
    # we import here to avoid errors when we are not execution on the raspberry pi
    from RaspberriModules.DataClasses.PowerModule import PowerModule

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
