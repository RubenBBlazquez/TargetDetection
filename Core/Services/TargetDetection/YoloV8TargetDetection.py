import os

import numpy as np
import pandas as pd

from Core.settings import directory_separator as slash
from Core.Services.TargetDetection.DetectionMethods import DetectionMethods
from ultralytics import YOLO
from ultralytics.engine.results import Results

class YoloV8TargetDetection(DetectionMethods):
    models_path = os.path.abspath(os.getcwd()) + f'{slash}models{slash}'

    def __init__(self, model_name, models_path = ''):
        self.model = YOLO(
            f'{models_path or self.models_path}{model_name}'
        )

    def predict(self, image) -> Results:
        return self.model.predict(image)

    def predict_and_filter_by_confidence(self, image: np.ndarray, confidence: float) -> pd.DataFrame:
        result: Results = self.model.predict(image)[0]
        tensor_labels = result.boxes.xywh
        tensor_labels = tensor_labels[result.boxes.conf > confidence]
        confidence = result.boxes.conf[result.boxes.conf > confidence]

        if len(tensor_labels) == 0:
            return pd.DataFrame(
                columns=['xcenter', 'ycenter', 'width', 'height', 'confidence']
            )

        return pd.DataFrame(
            np.hstack((tensor_labels, confidence.reshape(-1, 1))),
            columns=['xcenter', 'ycenter', 'width', 'height', 'confidence']
        )


    def train(self, yolo_file_data, epochs, batch_size):
        self.model.train(data=yolo_file_data, epochs=100)

    def save_model(self, model_name):
        pass
