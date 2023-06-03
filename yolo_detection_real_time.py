#!/usr/bin/python3
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import cv2
from picamera2 import MappedArray, Picamera2, Preview
import torch

normalSize = (640, 480)
lowresSize = (320, 240)

rectangles = []


def draw_targets(request):
    with MappedArray(request, "main") as m:
        if len(rectangles) > 0:
            for dimensions in rectangles[0]:
                x, y, w, h, precision, undefined = tuple(dimensions)

                if precision < 0.6:
                    continue

                cv2.rectangle(img=m.array, pt1=(int(x), int(y)), pt2=(int(x) + int(w), int(y) + int(h)), color=(0, 255, 0, 0))


# Model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gerion/TargetDetection/models/yolov5s/exp4/weights/best.pt', force_reload=True) 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cv2.startWindowThread()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": normalSize},
                                                 lores={"size": lowresSize, "format": "YUV420"})
picam2.configure(config)

picam2.start_preview(Preview.QTGL)
picam2.pre_callback = draw_targets

picam2.start()

while True:
    buffer = picam2.capture_buffer("lores")
    im = picam2.capture_array()
    results = model(im)
    results.show()
    rectangles = results.xywh
