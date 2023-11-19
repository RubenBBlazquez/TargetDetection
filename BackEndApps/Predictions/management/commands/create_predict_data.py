import ast
import json

import pandas as pd
from django.core.management.base import BaseCommand
import cv2

from BackEndApps.Predictions.models import GoodPredictions
import numpy as np

class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, *args, **options):
        predictions = GoodPredictions.objects.all()
        predict_data = []

        for prediction in predictions:
            print(1)
            list_of_lists = ast.literal_eval(prediction.predicted_image)
            image = np.array(list_of_lists, dtype=np.uint8)
            distances = json.loads(prediction.predicted_distances)
            right = distances['right']
            left = distances['left']
            center_target_x = distances['from_top_side_to_center']
            duty_cycle_per_cm = 0.5
            servo_position = prediction.servo_position

            if right > left:
                cm_to_center = left + center_target_x
                duty_cycle_position = servo_position + (cm_to_center * duty_cycle_per_cm)
                duty_cycle_position = duty_cycle_position if duty_cycle_position > 0 else 1
                print(f"1111, {duty_cycle_position}")
            else:
                cm_to_center = right + center_target_x
                duty_cycle_position = servo_position - (cm_to_center * duty_cycle_per_cm)
                duty_cycle_position = duty_cycle_position if duty_cycle_position > 0 else 0
                print(f"2222, {duty_cycle_position}")

            cv2.imshow('image', image)
            cv2.waitKey(300)

            predict_data.append({
                **distances,
                'actual_servo_position': servo_position,
                'target_servo_position': duty_cycle_position,
            })

            cv2.destroyAllWindows()

        pd.DataFrame(predict_data).to_csv('predict_data.csv', index=False)