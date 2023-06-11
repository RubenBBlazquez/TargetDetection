from celery import app
from Predictions.models import RawPredictionsData
import logging
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection

@app.shared_task
def check_prediction(*args):
    raw_data = RawPredictionsData(*args)
    logging.info('----------------------------------------------------------------------')
    logging.info(f'TASL RECIVED: yolo will predict with raw predictions data {raw_data}')
    yoloModel = YoloTargetDetection()


