import os
from typing import Any

import cv2
from django.core.management.base import BaseCommand

from BackEndApps.Predictions.ml_models import ServoMotorEnv, train_policy_network
from BackEndApps.Predictions.models import RawData
import pickle
from RaspberriModules.DataClasses.CustomPicamera import CustomPicamera
from RaspberriModules.DataClasses.ServoModule import ServoMovement, ServoManagement
from BackEndApps.Predictions.CeleryTasks import check_prediction, purge_celery
from datetime import datetime
import time
from enum import Enum
import numpy as np
import tensorflow as tf


class CameraType(Enum):
    USB = 'usb'
    CSI = 'csi'


class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usb_camera_port = 0
        self.usb_camera = cv2.VideoCapture(0)

    def get_image_from_camera(self, camera: Any, camera_type: CameraType):
        if camera_type == CameraType.CSI:
            return camera.capture_array()

        ret, image = camera.read()

        if type(image) != np.ndarray:
            self.usb_camera_port = 1 if self.usb_camera_port == 0 else 0
            camera = cv2.VideoCapture(
                self.usb_camera_port
            )
            ret, image = camera.read()

        return image

    def detection_with_celery(self):
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
            image = self.get_image_from_camera(picamera,  CameraType.CSI)
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

    @staticmethod
    def detection_with_ml_model(simulate_training=False):
        env = ServoMotorEnv()
        policy_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", input_shape=(env.observation_space.shape[0],)),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        train_policy_network(env, policy_net, tf.optimizers.Adam(learning_rate=0.01))
        policy_net.save('policy_net.h5')

    def add_arguments(self, parser):
        parser.add_argument('--use-celery', type=bool, default=False, help='path to the directory to train the models')

    def handle(self, *args, **options):
        use_celery = options.get('use_celery', False)

        if use_celery:
            self.detection_with_celery()
            return

        self.detection_with_ml_model()
