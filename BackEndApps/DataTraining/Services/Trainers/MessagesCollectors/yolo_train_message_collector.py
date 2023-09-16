from enum import Enum
from typing import Dict, List

import attr
import attrs

from BackEndApps.DataTraining.Services.Trainers.MessagesCollectors.base_train_message_collector import \
    TrainMessageCollector


class YoloTrainingPhase(Enum):
    TRAIN = 'starting training'
    TEST = 'testing'
    VAL = 'validation'
    FINISHED = 'epochs completed'


@attrs.define(auto_attribs=True)
class YoloTrainMessageCollector(TrainMessageCollector):
    important_messages: Dict[str, List[str]] = attr.ib(default={
        YoloTrainingPhase.TRAIN.name: [],
        YoloTrainingPhase.TEST.name: [],
        YoloTrainingPhase.VAL.name: []
    })

    def _check_if_training_phase_has_finished(self, stdout: str) -> bool:
        has_finished = YoloTrainingPhase.FINISHED.value in stdout.lower()

        if has_finished:
            self._training_phase = ''

        return has_finished

    def _set_phase_of_training_from_message(self, message: str) -> None:
        if YoloTrainingPhase.TRAIN.value in message.lower():
            self._training_phase = YoloTrainingPhase.TRAIN.name

        if YoloTrainingPhase.VAL.value in message.lower():
            self._training_phase = YoloTrainingPhase.VAL.name

        if YoloTrainingPhase.TEST.value in message.lower():
            self._training_phase = YoloTrainingPhase.TEST.name

    def collect_important_messages(self, stdout: str) -> None:
        if self._training_phase == '' or self._check_if_training_phase_has_finished(stdout):
            self._set_phase_of_training_from_message(stdout)

        if self._training_phase not in self.important_messages.keys():
            return

        self.important_messages[self._training_phase].append(stdout)