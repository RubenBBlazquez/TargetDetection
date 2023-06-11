import os

import torch

from Core.settings import directory_separator as slash
from Core.Services.TargetDetection.DetectionMethods import DetectionMethods
import yolov5


class YoloTargetDetection(DetectionMethods):
    models_path = os.path.abspath(os.getcwd()) + f'{slash}models{slash}yolov5s{slash}exp4{slash}weights{slash}'

    def __init__(self, model_name):
        self.model = yolov5.load(f'{self.models_path}{model_name}')
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

    def predict(self, image):
        return self.model(image)

    def train(self, yolo_file_data, epochs, batch_size):
        models_path = os.path.abspath(os.getcwd()) + '{slash}models{slash}'
        os.system(f'yolov5 train --img 640 --batch {batch_size} --epochs {epochs} --data {yolo_file_data}'
                  f' --weights {models_path}yolov5s.pt --save-period 1 --project {models_path}yolov5s')

    def save_model(self, model_name):
        yolov5.save(self.model, model_name)
