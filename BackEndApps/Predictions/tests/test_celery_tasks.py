import json
import pickle
import unittest.mock as mock
from datetime import datetime

import pytest
from bson import ObjectId
from freezegun import freeze_time
import numpy as np
import pandas as pd

from BackEndApps.Predictions.CeleryTasks import start_predictions_ok_actions, check_prediction
import cv2

from BackEndApps.Predictions.models import GoodPredictions, AllPredictions, RawPredictionData


@freeze_time("2020-01-01 12:00:00")
@pytest.mark.parametrize(
    'yolo_prediction, expected_prediction',
    [
        (
                pd.DataFrame({
                    'confidence': [0.9],
                }),
                1,
        ),
        # this test means that yolo predict 3 targets and 1 is ok, so the prediction is ok
        (
                pd.DataFrame({
                    'confidence': [0.4, 0.7, 0.8],
                }),
                1,
        ),
    ]
)
@mock.patch('BackEndApps.Predictions.CeleryTasks.celeryApp', autospec=True)
@mock.patch('BackEndApps.Predictions.CeleryTasks.YoloTargetDetection', autospec=True)
@mock.patch('BackEndApps.Predictions.CeleryTasks.start_predictions_ok_actions', autospec=True)
@pytest.mark.django_db
def test_check_prediction_with_prediction_ok(
        mock_start_predictions_ok_actions,
        yolo_target_detection_mock,
        celery_app_mock,
        yolo_prediction,
        expected_prediction,
):
    celery_app_mock.control.purge = mock.MagicMock()

    # Create a mock for the result of the predict method
    mock_predict_result = mock.MagicMock()
    mock_predict_result.pandas().xywh.__getitem__.return_value = yolo_prediction
    mock_model_instance = yolo_target_detection_mock.return_value
    mock_model_instance.predict.return_value = mock_predict_result

    mock_start_predictions_ok_actions.apply_async.return_value = None

    expected_image = np.array(cv2.imread('BackEndApps/Predictions/tests/assets/target_image.jpg'))
    check_prediction(pickle.dumps(expected_image), 0, datetime.now())
    result_prediction_object = AllPredictions.objects.get()

    launch_prediction_args = mock_start_predictions_ok_actions.apply_async.call_args_list[0]
    raw_prediction_data: RawPredictionData = launch_prediction_args[0][0]

    assert raw_prediction_data.image == pickle.dumps(expected_image)
    assert raw_prediction_data.prediction_id == str(result_prediction_object._id)
    assert result_prediction_object.prediction == expected_prediction


@freeze_time("2020-01-01 12:00:00")
@pytest.mark.parametrize(
    'yolo_prediction, expected_prediction',
    [
        (
                pd.DataFrame({
                    'confidence': [0.30],
                }),
                0,
        ),
        (
                pd.DataFrame({
                    'confidence': [0.49, 0.01, 0.1],
                }),
                0,
        ),
    ]
)
@mock.patch('BackEndApps.Predictions.CeleryTasks.celeryApp', autospec=True)
@mock.patch('BackEndApps.Predictions.CeleryTasks.YoloTargetDetection', autospec=True)
@mock.patch('BackEndApps.Predictions.CeleryTasks.start_predictions_ok_actions', autospec=True)
@pytest.mark.django_db
def test_check_prediction_with_prediction_not_ok(
        mock_start_predictions_ok_actions,
        yolo_target_detection_mock,
        celery_app_mock, yolo_prediction,
        expected_prediction,
):
    celery_app_mock.control.purge = mock.MagicMock()

    # Create a mock for the result of the predict method
    mock_predict_result = mock.MagicMock()
    mock_predict_result.pandas().xywh.__getitem__.return_value = yolo_prediction
    mock_model_instance = yolo_target_detection_mock.return_value
    mock_model_instance.predict.return_value = mock_predict_result

    expected_image = np.array(cv2.imread('BackEndApps/Predictions/tests/assets/target_image.jpg'))
    check_prediction(pickle.dumps(expected_image), 0, datetime.now())
    result_prediction_object = AllPredictions.objects.get()

    assert result_prediction_object.prediction == expected_prediction
    assert result_prediction_object.image == json.dumps(expected_image.tolist())

    mock_start_predictions_ok_actions.assert_not_called()


@freeze_time("2020-01-01 12:00:00")
@mock.patch('BackEndApps.Predictions.CeleryTasks.celeryApp', autospec=True)
@pytest.mark.django_db
def test_start_predictions_ok_actions(celery_app_mock):
    celery_app_mock.control.purge = mock.MagicMock()
    image = np.array(cv2.imread('BackEndApps/Predictions/tests/assets/target_image.jpg'))

    labels = pd.Series({
        'confidence': 0.99,
        'height': 100,
        'width': 100,
        'xcenter': 200,
        'ycenter': 200
    })

    expected_prediction_id = str(ObjectId())
    start_predictions_ok_actions(
        pickle.dumps(image),
        pickle.dumps(labels),
        expected_prediction_id,
        0,
        datetime.now()
    )

    prediction = GoodPredictions.objects.get(prediction_id=expected_prediction_id)

    assert prediction.prediction_id == expected_prediction_id
