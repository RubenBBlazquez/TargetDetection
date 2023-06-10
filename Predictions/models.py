from django.db import models
from datetime import datetime
from django.utils import timezone
from djongo import models as model_mg
from attrs import define, field
from datetime import datetime

@define(auto_attribs=True, frozen=True)
class RawPredictionsData:
    image: list = field()
    servo_position: int = field(default=0)
    date: datetime = field(default=datetime.now())

class Predictions(models.Model):
    image = models.TextField(db_column='image')
    labels = models.TextField(db_column='servo_position')
    prediction = models.BooleanField(db_column='prediction', default=False)
    checked = models.BooleanField(db_column='checked', default=False)
    date = models.DateTimeField(db_column='date', default=timezone.now)

    class Meta:
        db_table='Predictions'

