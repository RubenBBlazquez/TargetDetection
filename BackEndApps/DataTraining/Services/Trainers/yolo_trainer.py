import os
import shutil
import subprocess
import tempfile
from os import path

import attr

from BackEndApps.DataTraining.Services.Trainers.base_model_trainer import Trainer, NoTrainDataException, \
    TrainFailedException
from BackEndApps.DataTraining.Services.Uploaders.base_model_uploader import UploadModelFailedException
from Core.settings import directory_separator as slash


@attr.s(auto_attribs=True)
class YoloTrainer(Trainer):
    """
        This class is used to train a model with yoloV5.

        Parameters
        ----------
        batch_size: int
            This is the batch size that we want to use to train the model.
        epochs: int
            This is the number of epochs that we want to use to train the model.
        yolo_file_data: str
            This is the path to the file where the data.yaml is saved to train the model.
    """
    batch_size: int = attr.ib(default=15)
    epochs: int = attr.ib(default=50)
    yolo_file_data: str = attr.ib(default='')
    pretrained_model_path: str = attr.ib(default='')

    def _execute_training(self, cmd: str) -> bool:
        is_training_finished = False
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for line in iter(lambda: process.stdout.readline().decode('utf-8', errors='replace'), ''):
            line = line.replace('\n', '')
            print(line)

            is_training_finished = 'Results saved to' in line
            self.message_collector.collect_important_messages(line)

        return process.poll() != 0 and not is_training_finished

    def train(self) -> bool:
        model_path = tempfile.mkdtemp()

        if not self.yolo_file_data or not path.exists(self.yolo_file_data):
            raise NoTrainDataException(
                'You must provide a path to the file where the data.yaml is saved to train the model'
            )

        cmd = f'yolo detect train data={self.yolo_file_data} model={self.pretrained_model_path}'
        cmd += f' epochs={self.epochs} imgsz=640 device=mps batch={self.batch_size} workers=4'
        cmd += f' project={model_path}{slash}yolo{slash}'

        if not self._execute_training(cmd):
            raise TrainFailedException('The model failed to train')

        upload_result = self.model_uploader.upload(
            f'{model_path}{slash}yolo{slash}train{slash}weights{slash}best.pt'
        )

        if not upload_result:
            raise UploadModelFailedException('The model failed to upload')

        shutil.rmtree(model_path)

        return True