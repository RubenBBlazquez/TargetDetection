from celery import app
import pickle
from Predictions.models import RawPredictionsData
import logging

@app.shared_task
def check_prediction(raw_data: bytes):
    rawPredictionsData: RawPredictionsData = pickle.loads(raw_data)

    logging.info(f'celery working with raw predictions data {rawPredictionsData}')