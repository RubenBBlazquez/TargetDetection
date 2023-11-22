from abc import ABC, abstractmethod

import attr

from BackEndApps.DataTraining.Services.Trainers.MessagesCollectors.base_train_message_collector import \
    TrainMessageCollector, DummyTrainMessageCollector
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
        message_collector: TrainMessageCollector
            This is the message collector that we want to use to collect the messages from the training process.
        model_name: str
            This is the name of the model that we want to train.
    """
    model_uploader: ModelUploader
    message_collector: TrainMessageCollector = attr.ib(default=DummyTrainMessageCollector())
    model_name: str = attr.ib(default='yolo')

    @abstractmethod
    def train(self) -> bool:
        """
        This method is used to train a model.

        Returns
        -------
        bool: True if the model was trained successfully, False otherwise.

        Raises
        ------
        NoTrainDataException
            This exception is raised when there is no train data.
        TrainFailedException
            This exception is raised when the model failed to train.
        UploadModelFailedException
            This exception is raised when the model failed to upload.
        """
        raise NotImplementedError("You must implement this method train in a subclass")


@attr.s(auto_attribs=True)
class DummyTrainer(Trainer):
    """
        This class is to represent a dummy trainer.
    """

    def train(self, **kwargs) -> bool:
        raise Exception("You are using a dummy trainer")
