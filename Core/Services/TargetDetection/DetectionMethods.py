from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DetectionMethods(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> Any:
        """
        This method is used to predict the image.

        Parameters
        ----------
        image: np.ndarray
            This is the image that we want to predict.
        """
        raise NotImplementedError(
            'This method is not implemented in the child class. Please implement it in the child class.'
        )

    @abstractmethod
    def predict_and_filter_by_confidence(self, image: np.ndarray, confidence: float):
        """
        This method is used to predict the image and filter the results by confidence.

        Parameters
        ----------
        image: np.ndarray
            This is the image that we want to predict.
        confidence: int
            This is the confidence that we want to filter.
        """
        raise NotImplementedError(
            'This method is not implemented in the child class. Please implement it in the child class.'
        )

    @abstractmethod
    def train(self, yolo_file_data: str, epochs: int, batch_size: int):
        """
        This method is used to train the model.

        Parameters
        ----------
        yolo_file_data: str
            This is the path to the yolo file data.
        epochs: int
            This is the number of epochs that we want to train.
        batch_size: int
            This is the batch size that we want to use.
        """
        raise NotImplementedError(
            'This method is not implemented in the child class. Please implement it in the child class.'
        )

    @abstractmethod
    def save_model(self, model_name):
        """
        This method is used to save the model.

        Parameters
        ----------
        model_name: str
            This is the name of the model that we want to save.
        """
        raise NotImplementedError(
            'This method is not implemented in the child class. Please implement it in the child class.'
        )
