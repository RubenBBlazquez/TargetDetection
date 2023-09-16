import logging
from abc import ABC


class TrainerWriter(ABC):
    """
        This class is used to write the training results.
    """

    def write_on(self, **kwargs):
        """
            This method is used to write the training results based on the parameters.
        """
        raise NotImplementedError("You must implement this method write_on in a subclass")


class DummyTrainerWriter(TrainerWriter):
    """
        This class is to represent a dummy trainer writer.
    """

    def write_on(self):
        logging.error('You are using a dummy trainer writer so the results will not be written in any file.')
