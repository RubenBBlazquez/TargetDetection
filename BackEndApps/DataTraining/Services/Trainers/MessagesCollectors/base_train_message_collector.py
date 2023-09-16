import logging
from abc import ABC, abstractmethod
from enum import Enum

import attrs
import attr

from BackEndApps.DataTraining.Services.Trainers.Writers.base_trainer_writer import TrainerWriter, DummyTrainerWriter


class TrainingPhase(Enum):
    """
        This class is used to represent the phases of the training.
    """
    TRAIN = 'training'
    TEST = 'testing'
    VAL = 'validation'


@attrs.define(auto_attribs=True)
class TrainMessageCollector(ABC):
    """
        this class is used to collect messages from a training process

        Parameters
        ----------
        writer: TrainerWriter
            This is the writer used to write the training results.
        important_messages: dict
            This is the dictionary used to store the important messages of the training process.
        _training_phase: str
            This is the training phase of the training process.
    """

    writer: TrainerWriter = attr.ib(default=DummyTrainerWriter())
    important_messages: dict = attr.ib(default={
        TrainingPhase.TRAIN.name: [],
        TrainingPhase.TEST.name: [],
        TrainingPhase.VAL.name: []
    })
    _training_phase: str = attr.ib(default='')

    def _check_if_training_phase_has_finished(self, stdout: str) -> bool:
        """
            This method is used to check if the training phase has finished.

            Parameters
            ----------
            stdout: str
                This is the output message of the training process.

            Returns
            -------
                bool: True if the training phase has finished, False otherwise.
        """
        raise NotImplementedError("You must implement this method _check_if_training_phase_has_finished in a subclass")

    @abstractmethod
    def _set_phase_of_training_from_message(self, message: str) -> str:
        """
            This method is used to set the phase of the training from the message.

            Parameters
            ----------
            message: str
                This is the message to identify the phase of the training process.
        """
        raise NotImplementedError("You must implement this method _set_phase_of_training_from_message in a subclass")

    @abstractmethod
    def collect_important_messages(self, stdout: str) -> None:
        """
            This method is used to collect the important messages from the output of the training process.

            Parameters
            ----------
            stdout: str
                This is the output message of the training process.
        """
        raise NotImplementedError("You must implement this method _collect_important_messages in a subclass")


class DummyTrainMessageCollector(TrainMessageCollector):
    """
        This class is to represent a dummy train message collector.
    """

    def _check_if_training_phase_has_finished(self, stdout: str) -> bool:
        pass

    def _set_phase_of_training_from_message(self, message: str) -> str:
        return ''

    def collect_important_messages(self, stdout: str) -> None:
        logging.error('you are using a dummy train message collector so no messages will be collected')
        pass
