import logging
import pickle
import unittest.mock as mock
from datetime import datetime

import pytest
from bson import ObjectId
from freezegun import freeze_time
import numpy as np
import pandas as pd

from BackEndApps.Predictions.CeleryTasks import launch_prediction_action, check_prediction
import cv2

from BackEndApps.Predictions.models import GoodPredictions, AllPredictions, RawPredictionData


@freeze_time("2020-01-01 12:00:00")
@pytest.mark.parametrize(
    'yolo_prediction, expected_prediction, prediction_id',
    [
        (
                pd.DataFrame({
                    'confidence': [0.9],
                }),
                1,
                1
        ),
        (
                pd.DataFrame({
                    'confidence': [0.6, 0.7, 0.8],
                }),
                1,
                2
        ),
    ]
)
@mock.patch('BackEndApps.Predictions.CeleryTasks.celeryApp', autospec=True)
@mock.patch('BackEndApps.Predictions.CeleryTasks.YoloTargetDetection', autospec=True)
@mock.patch('BackEndApps.Predictions.CeleryTasks.launch_prediction_action', autospec=True)
@pytest.mark.django_db
def test_check_prediction_with_prediction_ok(
        launch_prediction_action_mock,
        yolo_target_detection_mock,
        celery_app_mock, yolo_prediction,
        expected_prediction,
        prediction_id
):
    celery_app_mock.control.purge = mock.MagicMock()

    # Create a mock for the result of the predict method
    mock_predict_result = mock.MagicMock()
    mock_predict_result.pandas().xywh.__getitem__.return_value = yolo_prediction
    mock_model_instance = yolo_target_detection_mock.return_value
    mock_model_instance.predict.return_value = mock_predict_result

    launch_prediction_action_mock.apply_async.return_value = None

    image = np.array(cv2.imread('BackEndApps/Predictions/tests/assets/target_image.jpg'))

    check_prediction(pickle.dumps(image), 0, datetime.now())

    prediction = AllPredictions.objects.get(prediction_id=prediction_id)

    assert prediction.prediction == expected_prediction

    launch_prediction_args = launch_prediction_action_mock.apply_async.call_args_list[0]
    raw_prediction_data: RawPredictionData = launch_prediction_args[0][0]

    assert raw_prediction_data.image == pickle.dumps(image)
    assert type(raw_prediction_data.prediction_id) is ObjectId


@freeze_time("2020-01-01 12:00:00")
@mock.patch('BackEndApps.Predictions.CeleryTasks.celeryApp', autospec=True)
@pytest.mark.django_db
def test_launch_prediction_action(celery_app_mock):
    celery_app_mock.control.purge = mock.MagicMock()
    image = np.array(cv2.imread('BackEndApps/Predictions/tests/assets/target_image.jpg'))

    labels = pd.Series({
        'confidence': 0.99,
        'height': 100,
        'width': 100,
        'xcenter': 200,
        'ycenter': 200
    })

    launch_prediction_action(pickle.dumps(image), pickle.dumps(labels), 0, 0, datetime.now())

    prediction = GoodPredictions.objects.get()
    breakpoint()
    breakpoint()