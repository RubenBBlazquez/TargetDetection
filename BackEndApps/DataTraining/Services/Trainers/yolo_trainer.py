import shutil
import subprocess
import tempfile
from enum import Enum
from os import path

import attr

from BackEndApps.DataTraining.Services.Trainers.base_model_trainer import Trainer, NoTrainDataException, \
    TrainFailedException
from BackEndApps.DataTraining.Services.Uploaders.base_model_uploader import DummyModelUploader, \
    UploadModelFailedException
from Core.settings import directory_separator as slash


class YoloTrainingPhase(Enum):
    """
        This class is used to represent the phases of the training.
    """
    TRAIN = 'starting training'
    TEST = 'test'
    VAL = 'val'


@attr.s(auto_attribs=True)
class YoloTrainer(Trainer):
    """
        This class is used to train a model with yoloV5.
    """
    _important_messages: dict = attr.ib(default={
        YoloTrainingPhase.TRAIN.name: [],
        YoloTrainingPhase.TEST.name: [],
        YoloTrainingPhase.VAL.name: []
    })
    _training_phase: str = attr.ib(default='')

    def _set_phase_of_training_from_message(self, message: str) -> str:
        """
            This method is used to set the phase of the training from the message.

            Parameters
            ----------
            message: str
                This is the message to identify the phase of the training process.

            Returns
            -------
            str: The phase of the training.
        """
        if YoloTrainingPhase.TRAIN.value in message:
            self._training_phase = YoloTrainingPhase.TRAIN.name

        if YoloTrainingPhase.VAL.value in message:
            self._training_phase = YoloTrainingPhase.TEST.name

        if YoloTrainingPhase.TEST.value in message:
            self._training_phase = YoloTrainingPhase.VAL.name

    def _collect_important_messages(self, stdout: str, phase_key: str) -> None:
        """
            This method is used to collect the important messages from the output of the training process.

            Parameters
            ----------
            stdout: str
                This is the output message of the training process.
        """
        message = stdout.split(' ')

        if len(message) > 1:
            self._important_messages[phase_key].append(message[1])

    def _execute_training(self, cmd: str) -> bool:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in iter(process.stdout.readline, ''):
            line = line.replace('\n', '')
            self._set_phase_of_training_from_message(line)
            print(line)

        return process.poll() == 0

    def train(self, **kwargs) -> bool:
        """
            This method is used to train a model with yoloV5.

            Parameters
            ----------
            kwargs: dict
                batch_size: int
                    This is the number of elements to train in groups each epoch.
                epochs: int
                    This is the number of epochs to train the yolo model.
                yolo_file_data: str
                    This is the path to the file data to train the model.

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
        model_path = tempfile.mkdtemp()
        batch_size = kwargs.get('batch_size', 15)
        epochs = kwargs.get('epochs', 50)
        yolo_file_data = kwargs.get('yolo_file_data', '')

        if not yolo_file_data or not path.exists(yolo_file_data):
            raise NoTrainDataException('You must provide a path to the file where the data is saved to train the model')

        cmd = f'yolov5 train --img 640 --batch {batch_size} --epochs {epochs} --data {yolo_file_data}'
        cmd += f' --weights {model_path}{slash}yolov5s.pt --save-period 1 --project {model_path}{slash}yolov5s'

        if not self._execute_training(cmd):
            raise TrainFailedException('The model failed to train')

        upload_result = self.model_uploader.upload(
            f'{model_path}{slash}yolov5s{slash}exp4{slash}weights{slash}best.pt'
        )

        if not upload_result:
            raise UploadModelFailedException('The model failed to upload')

        shutil.rmtree(model_path)

        return True


if __name__ == '__main__':
    trainer = YoloTrainer(DummyModelUploader())
    trainer.train(
        **{'batch_size': 15, 'epochs': 50,
           'yolo_file_data': 'C:\\Users\\rbblazquez\\Desktop\\TargetDetectionDatasets\\dataset1\\data.yaml'}
    )
