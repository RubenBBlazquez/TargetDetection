from celery import shared_task

from Predictions.models import RawPredictionsData

@shared_task
def check_prediction(raw_data: RawPredictionsData):
    image, servo_angle, date = raw_data