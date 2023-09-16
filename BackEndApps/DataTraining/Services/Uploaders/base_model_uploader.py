import logging
from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum
from typing import Any

import attr


class UploadModelFailedException(Exception):
    """
        This class is used to represent an exception when the model failed to upload.
    """
    pass


class UploadServiceTypes(Enum):
    """
        This class is used to define the types of services that we can use to upload the trained models.
    """
    AWS = 'aws'
    GOOGLE = 'google'
    LOCAL = 'local'


@attr.s(auto_attribs=True)
class ModelUploader(ABC):
    """
        This class is used to upload a model to any service.
    """
    service_type: str = attr.ib(default=UploadServiceTypes.LOCAL.value)

    @abstractmethod
    def upload(self, model_directory: str) -> bool:
        """
        This method is used to upload the model to any service such as cloud or local.

        Parameters
        ----------
        model_directory: str
            This is the directory where the model is saved.

        Returns
        -------
        bool: True if the model was uploaded successfully, False otherwise.
        """
        raise NotImplementedError("You must implement this method upload in a subclass")


@attr.s(auto_attribs=True)
class DummyModelUploader(ModelUploader):
    """
        This class is to represent a dummy uploader.
    """

    def upload(self, model: Any) -> bool:
        logging.error('You are using a dummy uploader, the model will not be uploaded to any service.')

        return False
