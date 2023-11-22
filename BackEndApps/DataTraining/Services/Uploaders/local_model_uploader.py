import os
from typing import Any
import shutil

import attr

from BackEndApps.DataTraining.Services.Uploaders.base_model_uploader import ModelUploader


@attr.s(auto_attribs=True)
class LocalModelUploader(ModelUploader):
    """
        This class is to represent a dummy uploader.
    """
    new_path: str = attr.ib(default='')

    def upload(self, model: Any) -> bool:
        if not self.new_path:
            raise Exception('You must provide a path to upload the model')

        shutil.copy(model, self.new_path)

        return True