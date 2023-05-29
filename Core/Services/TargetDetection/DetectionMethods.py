from abc import ABC, abstractmethod


class DetectionMethods(ABC):
    @abstractmethod
    def predict(self, image):
        pass

    @abstractmethod
    def train(self, yolo_file_data, epochs, batch_size):
        pass

    @abstractmethod
    def save_model(self, model_name):
        pass
