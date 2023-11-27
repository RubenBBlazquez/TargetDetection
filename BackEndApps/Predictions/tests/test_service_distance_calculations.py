import json
import logging
import unittest.mock
from unittest.mock import MagicMock, call
import cv2
import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from BackEndApps.Predictions.services.DistanceCalculations import DistanceCalculations, CM_IN_PIXELS


@pytest.fixture(scope='function')
def test_image(request: SubRequest):
    image_path = request.param
    image = np.array(cv2.imread(image_path))

    return image


mark_different_images = pytest.mark.parametrize(
    'test_image, labels',
    [
        (
            'BackEndApps/Predictions/tests/assets/target_image.jpg',
            pd.Series({
                'confidence': 0.99,
                'height': 100,
                'width': 100,
                'xcenter': 100,
                'ycenter': 100
            })
        ),
        (
            'BackEndApps/Predictions/tests/assets/target_image_2.jpg',
            pd.Series({
                'confidence': 0.99,
                'height': 100,
                'width': 100,
                'xcenter': 80,
                'ycenter': 20
            })
        )
    ],
    indirect=['test_image']
)


@mark_different_images
def test_calculate_left_distance(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    line_to_left = distance_calculations.line_to_left_side

    result_cm = distance_calculations.get_distance_in_cm(line_to_left)
    expected_cm = abs(round(int(labels.xcenter - labels.width / 2) / CM_IN_PIXELS, 3))

    assert result_cm == expected_cm


@mark_different_images
def test_calculate_upper_distance(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    line_to_up = distance_calculations.line_to_upper_side

    result_cm = distance_calculations.get_distance_in_cm(line_to_up)
    expected_cm = abs(round(int(labels.ycenter - labels.height / 2) / CM_IN_PIXELS, 3))

    assert result_cm == expected_cm


@mark_different_images
def test_calculate_right_distance(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    line_to_right = distance_calculations.line_to_right_side

    result_cm = distance_calculations.get_distance_in_cm(line_to_right)
    x1 = labels.xcenter + labels.width / 2
    expected_cm = abs(round(int(test_image.shape[1] - x1) / CM_IN_PIXELS, 3))

    assert result_cm == expected_cm


@mark_different_images
def test_calculate_bottom_distance(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    line_to_bottom = distance_calculations.line_to_bottom_side

    result_cm = distance_calculations.get_distance_in_cm(line_to_bottom)
    y1 = labels.ycenter + labels.height / 2
    expected_cm = abs(round(int(test_image.shape[0] - y1) / CM_IN_PIXELS, 3))

    assert result_cm == expected_cm

@mark_different_images
def test_calculate_max_width_distance(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    line_to_max_width = distance_calculations.line_to_max_width

    result_cm = distance_calculations.get_distance_in_cm(line_to_max_width)
    expected_cm = round(test_image.shape[1]/CM_IN_PIXELS, 3)

    assert result_cm == expected_cm

@mark_different_images
def test_calculate_max_height_distance(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    line_to_max_height = distance_calculations.line_to_max_height

    result_cm = distance_calculations.get_distance_in_cm(line_to_max_height)
    expected_cm = round(test_image.shape[0]/CM_IN_PIXELS, 3)
    assert result_cm == expected_cm

@mark_different_images
@unittest.mock.patch('BackEndApps.Predictions.services.DistanceCalculations.cv2')
def test_draw_lines_into_image(cv2_mock: MagicMock, test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    line_to_up = distance_calculations.line_to_upper_side
    line_to_left = distance_calculations.line_to_left_side
    lines = [line_to_up, line_to_left]

    distance_calculations.draw_lines_into_image(1, lines)

    expected_line_calls = [
        call(test_image, line_to_up.pt1, line_to_up.pt2, (255, 0, 0), 1),
        call(test_image, line_to_left.pt1, line_to_left.pt2, (255, 0, 0), 1)
    ]
    result_cv2_line_method_call_args_list = cv2_mock.line.call_args_list

    distance_line_up = distance_calculations.get_distance_in_cm(line_to_up)
    coords_line_up = distance_calculations.calculate_coords_text_cm(line_to_up)
    distance_line_left = distance_calculations.get_distance_in_cm(line_to_left)
    coords_line_left = distance_calculations.calculate_coords_text_cm(line_to_left)
    font = cv2_mock.FONT_HERSHEY_SIMPLEX
    line_type = cv2_mock.LINE_AA

    expected_put_text_calls = [
        call(test_image, f'{distance_line_up} cm', coords_line_up, font, 0.5, (255, 0, 0), 1, line_type),
        call(test_image, f'{distance_line_left} cm', coords_line_left, font, 0.5, (255, 0, 0), 1, line_type),
    ]
    result_cv2_put_text_method_call_args_list = cv2_mock.putText.call_args_list

    assert list(result_cv2_line_method_call_args_list) == expected_line_calls
    assert list(result_cv2_put_text_method_call_args_list) == expected_put_text_calls


@mark_different_images
def test_draw_lines_showing_images(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    distance_calculations.draw_lines_into_image(10000)

@mark_different_images
def test_get_all_distances(test_image, labels):
    distance_calculations = DistanceCalculations.create_from(test_image, labels)
    distances = distance_calculations.get_all_distances()

    assert list(distances.index) == ['left', 'top', 'right', 'bottom', 'from_top_side_to_center', 'from_bottom_side_to_center', 'width', 'height']


@pytest.mark.skip(reason='This test is only for manual testing')
def test_with_real_image():
    predictions = pd.read_csv('TargetDetection/Predictions/tests/assets/predictions.csv')

    for index, row in predictions.iterrows():
        image = np.array(json.loads(row['original_image']))
        converted_image = cv2.convertScaleAbs(image)
        labels = eval(json.loads(row['labels']))
        # x = (int(labels['xcenter']) - int(labels['width']) // 2, int(labels['ycenter']) - int(labels['height']) // 2)
        # y = (int(labels['xcenter']) + int(labels['width']) // 2, int(labels['ycenter']) + int(labels['height']) // 2)
        # cv2.rectangle(converted_image, x, y, (255, 0, 0), 3)
        # cv2.imshow('image', converted_image)

        cv2.waitKey(3000)

        logging.error(f'Image: {labels}')

    assert False