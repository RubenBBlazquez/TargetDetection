import logging

import cv2
import numpy as np
import pandas as pd

from Predictions.services.DistanceCalculations import DistanceCalculations


def test_calculate_left_distance():
    image = np.array(cv2.imread('Predictions/tests/assets/target_image.jpg'))
    labels = pd.Series({
        'confidence': 0.99,
        'height': 100,
        'width': 100,
        'xcenter':100,
        'ycenter': 100
    })
    distance_calculations = DistanceCalculations.create_from(image, labels)
    line_to_left = distance_calculations.distance_to_left()
    line_to_up = distance_calculations.distance_to_up()
    distance_calculations.draw_lines_into_image([line_to_left, line_to_up])
    logging.error(f"distance_calculations.distance_to_left: {line_to_up}")

    assert False