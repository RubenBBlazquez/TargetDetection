import os
import cv2
from django.core.management.base import BaseCommand
from BackEndApps.Predictions.models import RawData
import pickle
from RaspberriModules.DataClasses.CustomPicamera import CustomPicamera
from RaspberriModules.DataClasses.ServoModule import ServoMovement
from BackEndApps.Predictions.CeleryTasks import check_prediction, purge_celery
from datetime import datetime
import time
from enum import Enum
import numpy as np

class CameraType(Enum):
    USB = 'usb'
    CSI = 'csi'
def enumerate_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.picamera = CustomPicamera()
        self.picamera.start()
        self.usb_camera_port = 0
        self.usb_camera = cv2.VideoCapture(self.usb_camera_port)
    
    def get_image_from_camera(self, camera_type: CameraType, init_cam=True):
        if camera_type == CameraType.CSI:
            return self.picamera.capture_array()    
        
        ret, image = self.usb_camera.read()

        if type(image) != np.ndarray:
            self.usb_camera_port = 1 if self.usb_camera_port == 0 else 0
            self.usb_camera = cv2.VideoCapture(
                self.usb_camera_port
            )
            ret, image = self.usb_camera.read()

        return image
        
    def handle(self, *args, **options):
        frames = 0
        angle = 1
        gpin_horizontal_servo = int(os.getenv('X_SERVO_PIN'))
        increment = 1
        servo_movements = 0
        cv2.startWindowThread()

        while True:
            frames += 1
            image = self.get_image_from_camera(CameraType.CSI, True)
            cv2.imshow("CSI Camera", image)
            cv2.waitKey(1)       
            is_shoot_in_progress = os.path.exists('RaspberriModules/assets/shoot_in_progress.tmp')

            if frames < 15 or is_shoot_in_progress:
                continue
            
            if angle < 0:
                angle = 1

            second_camera_image = self.get_image_from_camera(CameraType.USB, True)
            cv2.imshow("USB Camera", second_camera_image)

            servo = ServoMovement(gpin_horizontal_servo, angle)
            servo.default_move()
            time.sleep(0.1)

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
                purge_celery.apply_async(('check_predictions', ), queue='purge_data', ignore_result=True, prority=1)

            frames = 0

        cv2.destroyAllWindows()