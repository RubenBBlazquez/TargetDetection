#!/usr/bin/python3
from pprint import pprint
from time import sleep
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2
from picamera2 import MappedArray, Picamera2, Preview
import torch
from torch import tensor
import pandas as pd
from DataClasses.ServoModule import ServoMovement

normalSize = (400, 500)

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gerion/TargetDetection/models/yolov5s/exp4/weights/best.pt', force_reload=False) 
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cv2.startWindowThread()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": normalSize}, controls={'FrameRate': 30})
picam2.configure(config)
picam2.start_preview(Preview.QTGL)
picam2.start()

def get_valid_predictions(predictions: pd.DataFrame) -> List[tensor]:
    if predictions.empty:
        return predictions
    
    valid_predictions = predictions.loc[
        predictions['confidence'] >= 0.65,
        ['xcenter', 'ycenter', 'width', 'height']
    ]

    return valid_predictions.astype(np.int32)

def predict_last_images(last_actions: List[np.array]):
    for index, action in enumerate(last_actions[::-1]):
        image, servo_module = action
        results = model(image)
        predictions = results.pandas().xywh[0]
        valid_predictions = get_valid_predictions(predictions)

        print(index, valid_predictions)
        if valid_predictions.size > 0:
            servo_module.default_move()
            break


frames = 0
last_actions = []
angle = 0
gpin_horizontal_servo = 13
increment = 2.5

while True:
    frames += 1
    image = picam2.capture_array()
    
    if frames >= 15:
        print('---- add one more movement ----')
        servo = ServoMovement(gpin_horizontal_servo, angle)
        last_actions.append((image, servo))

        if len(last_actions) == 4:
            servo.stop()
            predict_last_images(last_actions)
            last_actions = []
            increment = -increment
        
        angle += increment

        if angle < 0:
            angle = 0

        frames = 0


