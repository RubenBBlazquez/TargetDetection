import os
import cv2
from django.core.management.base import BaseCommand
from BackEndApps.Predictions.models import RawData
import pickle
from RaspberriModules.DataClasses.CustomPicamera import CustomPicamera
from RaspberriModules.DataClasses.ServoModule import ServoMovement
from BackEndApps.Predictions.CeleryTasks import check_prediction, purge_celery
from datetime import datetime

class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.picamera = CustomPicamera()
        self.picamera.start_camera()
    
    def handle(self, *args, **options):
        frames = 0
        angle = 0
        gpin_horizontal_servo = 11
        increment = 2
        servo_movements = 0
        cv2.startWindowThread()

        while True:
            frames += 1
            image = self.picamera.capture_array()
            cv2.imshow("Camera", image)
            cv2.waitKey(1)       
            is_shoot_in_progress = os.path.exists('RaspberriModules/assets/shoot_in_progress.tmp')

            if frames < 15 or is_shoot_in_progress:
                continue
            
            if angle < 1:
                angle = 3

            servo = ServoMovement(gpin_horizontal_servo, angle)
            servo.stop()

            servo_movements += 1
            raw_data = RawData(image=pickle.dumps(image), servo_position=angle, date=datetime.now())
            
            check_prediction.apply_async(raw_data, queue='YoloPredictions', ignore_result=True, prority=1)

            if servo_movements % 5 == 0:
                increment = -increment
                
            angle += increment
            
            # we check if servo do all possible movements and clean unnecessary tasks
            if servo_movements % 8 == 0:
                print('purging tasks')
                purge_celery.apply_async(queue='YoloPredictions', ignore_result=True, prority=1)

            frames = 0

cv2.destroyAllWindows()