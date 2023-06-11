from typing import NamedTuple
from django.db import models
from datetime import datetime
from django.utils import timezone
from djongo import models as model_mg
from attrs import define, field
from datetime import datetime
import numpy as np

class RawPredictionsData(NamedTuple):
    image: list
    servo_position: int
    date: datetime

    def __str__(self) -> str:
        return f'image: {len(self.image)}, servo_angle: {self.servo_position}, date: {self.date}'

class Predictions(models.Model):
    image = models.TextField(db_column='image')
    labels = models.TextField(db_column='servo_position')
    prediction = models.BooleanField(db_column='prediction', default=False)
    checked = models.BooleanField(db_column='checked', default=False)
    date = models.DateTimeField(db_column='date', default=timezone.now)

    class Meta:
        db_table='Predictions'

