import os

from Core.settings import directory_separator as slash
from Core.Services.TargetDetection.DetectionMethods import DetectionMethods
from ultralytics import YOLO
from ultralytics.engine.results import Results

class YoloTargetDetection(DetectionMethods):
    models_path = os.path.abspath(os.getcwd()) + f'{slash}models{slash}'

    def __init__(self, model_name):
        self.model = YOLO(f'{self.models_path}{model_name}')

    def predict(self, image) -> Results:
        return self.model.predict(image)

    def train(self, yolo_file_data, epochs, batch_size):
        self.model.train(data=yolo_file_data, epochs=100)

    def save_model(self, model_name):
        pass
