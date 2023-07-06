from typing import NamedTuple

from attr import define
from django.db import models
from django.utils import timezone
from datetime import datetime


class RawData(NamedTuple):
    image: bytes
    servo_position: int
    date: datetime

    def __str__(self) -> str:
        return f'image: {len(self.image)}, servo_angle: {self.servo_position}, date: {self.date}'


class RawPredictionData(NamedTuple):
    image: bytes
    labels: bytes
    servo_position: int
    date: datetime

    def __str__(self) -> str:
        return f'image: {len(self.image)}, servo_angle: {self.servo_position}, date: {self.date}'


@define(auto_attribs=True)
class CleanPredictionData:
    original_image: str
    labels: str
    predicted_image: str
    servo_position: int


class Predictions(models.Model):
    original_image = models.TextField(db_column='image')
    predicted_image = models.TextField(db_column='predicted_image')
    labels = models.TextField(db_column='predicted_labels')
    servo_position = models.TextField(db_column='servo_position')
    checked = models.BooleanField(db_column='checked', default=False)
    date = models.DateTimeField(db_column='date', default=timezone.now)

    @classmethod
    def create_from(cls, data: CleanPredictionData):
        return cls(
            original_image=data.original_image,
            predicted_image=data.predicted_image,
            labels=data.labels,
            servo_position=data.servo_position
        )

    class Meta:
        db_table = 'Predictions'
