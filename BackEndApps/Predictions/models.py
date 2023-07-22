from __future__ import annotations
from typing import NamedTuple

from attr import define
from django.db import models
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
    prediction_id: str
    servo_position: int
    date: datetime

    def __str__(self) -> str:
        return f'image: {len(self.image)}, servo_angle: {self.servo_position}, date: {self.date}'


@define(auto_attribs=True)
class CleanPredictionData:
    original_image: str
    labels: str
    predicted_image: str
    predicted_distances: str
    servo_position: int
    prediction_id: int


class AllPredictions(models.Model):
    """
        This class is used to save all the predictions, even if the prediction is good or bad.
    """
    id = models.AutoField(primary_key=True)
    image = models.TextField(db_column='image', default='')
    prediction = models.BooleanField(db_column='prediction', default=False)
    confidence = models.FloatField(db_column='confidence', default=0.0)

    class Meta:
        db_table = 'AllPredictions'


class GoodPredictions(models.Model):
    """
        This class is used to save the predictions in which we have calculated the distances,
        and we have shot to the target.
    """
    id = models.AutoField(primary_key=True)
    original_image = models.TextField(db_column='image', default='')
    predicted_image = models.TextField(db_column='predicted_image', default='')
    labels = models.TextField(db_column='predicted_labels', default='')
    predicted_distances: models.TextField(db_column='predicted_distances', default='')
    servo_position = models.IntegerField(db_column='servo_position', default=0)
    checked = models.BooleanField(db_column='checked', default=False)
    date = models.DateTimeField(db_column='date', default=datetime.now())
    prediction_id = models.IntegerField(db_column='prediction_id', default=0)

    @classmethod
    def create_from(cls, data: CleanPredictionData) -> GoodPredictions:
        return cls(
            original_image=data.original_image,
            predicted_image=data.predicted_image,
            labels=data.labels,
            servo_position=data.servo_position
        )

    def __str__(self):
        return f'original_image: {self.original_image}, ' \
               f'predicted_image: {self.predicted_image},' \
               f' labels: {self.labels}, servo_position: {self.servo_position},' \
               f' checked: {self.checked}, date: {self.date}'

    class Meta:
        db_table = 'GoodPredictions'
