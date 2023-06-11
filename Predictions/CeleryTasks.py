from celery import app
from datetime import datetime
from Predictions.models import RawPredictionData
import logging
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection
import pickle
import os
import numpy as np
import pandas as pd

@app.shared_task
def check_prediction(*args):
    image_bytes, servo_position, date = args
    image = pickle.loads(image_bytes)

    logging.info('----------------------------------------------------------------------')
    logging.info(f'TASK RECEIVED: {datetime.now()}')

    if datetime.now().day != date.day:
        logging.info(f'TASK REJECTED: DATE DAY ({date.day}) ARGUMENT IS DIFFERENT FROM ACTUAL DATE ({datetime.now().day})')

        return
    
    model = YoloTargetDetection(os.getenv('YOLO_MODEL_NAME'))
    result_prediction = model.predict(image)
    predicted_labels = result_prediction.pandas().xywh[0]
    predicted_labels = predicted_labels[predicted_labels['confidence'] > 60]

    logging.info(f'Predicted Labels: \n {predicted_labels}')

    if not predicted_labels.empty:
        predicted_labels.apply(
            lambda labels: 
                launch_prediction_action.apply_async(
                    RawPredictionData(pickle.dumps(image), pickle.dumps(labels), servo_position, datetime.now()),
                    ignore_result=True
                ), 
            axis=1
        )

    logging.info('----------------------------------------------------------------------')


@app.shared_task
def launch_prediction_action(*args):
    image_bytes, labels_bytes, servo_position, date = args
    image: np.ndarray = pickle.loads(image_bytes)
    labels: pd.Series = pickle.loads(labels_bytes)

    if datetime.now().day != date.day:
        logging.info(f'TASK REJECTED: DATE DAY ({date.day}) ARGUMENT IS DIFFERENT FROM ACTUAL DATE ({datetime.now().day})')

        return

    #os.system('python -m celery -A Core purge -f')



