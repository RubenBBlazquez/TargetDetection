import pickle
import unittest.mock as mock
from datetime import datetime
from freezegun import freeze_time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from Predictions.CeleryTasks import launch_prediction_action
import cv2


@freeze_time("2020-01-01 12:00:00")
@mock.patch('Predictions.CeleryTasks.celeryApp', autospec=True)
def test_launch_prediction_action(celery_app_mock):
    celery_app_mock.control.purge = mock.MagicMock()

    image = np.array(plt.imread('tests/images/target_image.jpg'))
    image = image.reshape((500, 600, 3))
    labels = pd.Series({
        'confidence': 0.99,
        'height': 100,
        'width': 100,
        'xcenter': 100,
        'ycenter': 100
    })

    launch_prediction_action((pickle.dumps(image), pickle.dumps(labels), 0, datetime.now()))
