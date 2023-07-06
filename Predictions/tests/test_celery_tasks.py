import logging
import os
import pickle
import unittest.mock as mock
from datetime import datetime

import pytest
from freezegun import freeze_time
import numpy as np
import pandas as pd

from Predictions.CeleryTasks import launch_prediction_action
import cv2


@freeze_time("2020-01-01 12:00:00")
@mock.patch('Predictions.CeleryTasks.celeryApp', autospec=True)
@pytest.mark.django_db
def test_launch_prediction_action(celery_app_mock):
    celery_app_mock.control.purge = mock.MagicMock()
    image = np.array(cv2.imread('Predictions/tests/assets/target_image.jpg'))

    labels = pd.Series({
        'confidence': 0.99,
        'height': 100,
        'width': 100,
        'xcenter': 200,
        'ycenter': 200
    })

    launch_prediction_action(pickle.dumps(image), pickle.dumps(labels), 0, datetime.now())
