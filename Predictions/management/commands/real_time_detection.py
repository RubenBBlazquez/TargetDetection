from attrs import asdict
from django.core.management.base import BaseCommand
from Predictions.models import RawPredictionsData
import pickle
from RaspberriModules.DataClasses.CustomPicamera import CustomPicamera
from RaspberriModules.DataClasses.ServoModule import ServoMovement
from Predictions.CeleryTasks import check_prediction
from datetime import datetime

class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.picamera = CustomPicamera()
        self.picamera.start_camera()

    def handle(self, *args, **options):
        frames = 0
        angle = 0
        gpin_horizontal_servo = 13
        increment = 3
        servo_movements = 0

        while True:
            frames += 1
            image = self.picamera.capture_array()
            
            if frames < 15:
                continue
            
            servo = ServoMovement(gpin_horizontal_servo, angle if angle > 1 else 1)
            servo.stop()

            servo_movements += 1
            raw_data = RawPredictionsData(image=image.tolist(), servo_position=angle, date=datetime.now())

            check_prediction.apply_async(raw_data, ignore_result=True)

            if servo_movements == 4:
                increment = -increment
                servo_movements = 0
                
            angle += increment

            frames = 0