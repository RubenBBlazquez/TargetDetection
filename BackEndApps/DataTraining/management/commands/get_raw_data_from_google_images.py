import os

import cv2
import numpy as np
from django.core.management import BaseCommand
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection

PATH_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


class Command(BaseCommand):
    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        print(YoloTargetDetection('best.pt').predict(open(f'{PATH_DIR}\\descarga.jpg', 'rb')))
        pass


if __name__ == '__main__':
    file = open(f'{PATH_DIR}\\descarga6.jpg', 'rb')
    array = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(array, 1)
    prediction = YoloTargetDetection('best.pt').predict(image)

    prediction.show()
