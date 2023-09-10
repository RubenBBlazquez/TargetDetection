from abc import ABC, abstractmethod

import attr

from BackEndApps.DataTraining.Services.Uploaders.base_model_uploader import ModelUploader


class NoTrainDataException(Exception):
    """
        This class is used to represent an exception when there is no train data.
    """
    pass


class TrainFailedException(Exception):
    """
        This class is used to represent an exception when the model failed to train.
    """
    pass


@attr.s(auto_attribs=True)
class Trainer(ABC):
    """
        This class is used to train a model.

        Parameters
        ----------
        model_uploader: ModelUploader
            This is the model uploader that we want to use to upload the model after we train it.
    """
    model_uploader: ModelUploader

    @abstractmethod
    def train(self, **kwargs) -> bool:
        """
        This method is used to train a model.

        Parameters
        ----------
        kwargs: dict
            This is the dictionary with the parameters to train the model.

        Returns
        -------
        bool: True if the model was trained successfully, False otherwise.
        """
        raise NotImplementedError("You must implement this method train in a subclass")


@attr.s(auto_attribs=True)
class DummyTrainer(Trainer):
    """
        This class is to represent a dummy trainer.
    """

    def train(self, **kwargs) -> bool:
        raise Exception("You are using a dummy trainer")
