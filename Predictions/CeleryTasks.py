from celery import app
from Core.Celery.Celery import app as celeryApp
from datetime import datetime
from Predictions.models import RawPredictionData
import logging
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection
import pickle
import os
import numpy as np
import pandas as pd
import cv2

@app.shared_task
def purge_celery():
    """
    This task is used to clean all tasks in celery queue
    """
    inspection = celeryApp.control.inspect()
    tasks = inspection.reserved()

    if len(tasks) > 15:
        celeryApp.control.purge()

@app.shared_task
def check_prediction(*args):
    """
        This task is used to check if prediction is correct, when we receive a prediction from YoloTargetDetection.
        If prediction is correct, we launch a task to manage the prediction and save on database.

        Parameters:
        -----------
        *args: tuple
            This tuple contains the following arguments:
                - image_bytes: bytes
                    This argument contains the image in bytes format.
                - servo_position: int
                    This argument contains the servo position when the image was taken.
                - date: datetime
                    This argument contains the date when the image was taken.
    """
    image_bytes, servo_position, date = args
    image = pickle.loads(image_bytes)

    logging.info('----------------------------------------------------------------------')
    logging.info(f'TASK (check_prediction) RECEIVED: {datetime.now()}')

    if datetime.now().day != date.day:
        logging.info(f'TASK (check_prediction) REJECTED: DATE DAY ({date.day}) ARGUMENT IS DIFFERENT FROM ACTUAL DATE ({datetime.now().day})')

        return
    
    model = YoloTargetDetection(os.getenv('YOLO_MODEL_NAME'))
    result_prediction = model.predict(image)
    predicted_labels = result_prediction.pandas().xywh[0]
    
    logging.info(predicted_labels['confidence'] > 0.60)

    predicted_labels = predicted_labels[predicted_labels['confidence'] > 0.60]
    
    logging.info(f'Predicted Labels: \n {predicted_labels}')

    if not predicted_labels.empty:
        predicted_labels.apply(
            lambda labels: 
                launch_prediction_action.apply_async(
                    RawPredictionData(pickle.dumps(image), pickle.dumps(labels), servo_position, datetime.now()),
                    ignore_result=True,
                    queue='YoloPredictions',
                    priority=10
                ), 
            axis=1
        )

@app.shared_task
def launch_prediction_action(*args):
    """
        This task is used to launch a task to manage the prediction and save on database.

        Parameters:
        -----------
        *args: tuple
            This tuple contains the following arguments:
                - image_bytes: bytes
                    This argument contains the image in bytes format.
                - labels_bytes: bytes
                    This argument contains the labels from the prediction in bytes format.
                - servo_position: int
                    This argument contains the servo position when the image was taken.
                - date: datetime
                    This argument contains the date when the image was taken.
    """
    image_bytes, labels_bytes, servo_position, date = args
    image: np.ndarray = pickle.loads(image_bytes)
    labels: pd.Series = pickle.loads(labels_bytes)

    if datetime.now().day != date.day:
        logging.info(f'TASK (launch_prediction_action) REJECTED: DATE DAY ({date.day}) ARGUMENT IS DIFFERENT FROM ACTUAL DATE ({datetime.now().day})')

        return

    image = cv2.rectangle(
        image, 
        (int(labels.xcenter), int(labels.ycenter)), 
        (int(labels.xcenter) + int(labels.width), int(labels.ycenter) + int(labels.width)),
        (36,255,12), 
        1
    )

    cv2.imshow(image)
        
    os.system('python -m celery -A Core purge -f')



