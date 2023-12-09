import os
import pickle
import time
from datetime import datetime

import cv2

from BackEndApps.Predictions.ml_models.real_turret_env import get_image_from_camera, CameraType
from BackEndApps.Predictions.models import RawData
from BackEndApps.Predictions.CeleryTasks import check_prediction, purge_celery

def real_time_detection_with_celery():
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
        raw_data = RawData(image=pickle.dumps(image), servo_position=angle, date=datetime.now())

        check_prediction.apply_async(raw_data, queue='check_predictions', ignore_result=True, prority=1)

        if servo_movements % 10 == 0:
            increment = -increment

        angle += increment
        # we check if servo do all possible movements and clean unnecessary tasks
        if servo_movements % 20 == 0:
            print('purging tasks')
            servo_movements = 0
            purge_celery.apply_async(('check_predictions',), queue='purge_data', ignore_result=True, prority=1)

        frames = 0

    cv2.destroyAllWindows()