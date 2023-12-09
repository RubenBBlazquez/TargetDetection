import cv2
import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from Core.Services.TargetDetection.YoloV8TargetDetection import YoloV8TargetDetection


@pytest.fixture
def yolo_model():
    return YoloV8TargetDetection('yolo_model_test.pt', 'Core/tests/Services/assets/')

@pytest.fixture
def image_with_target():
    return

@pytest.fixture
def image_without_target():
    return

def test_yolo_v8_target_detection_predict(yolo_model):
    pass

@pytest.mark.parametrize(
    'image, expected_result',
    [
        (
            np.array(cv2.imread('Core/tests/Services/assets/target_image.jpg')),
            pd.DataFrame(
                [{
                    "xcenter": 126.41,
                    "ycenter": 92.66,
                    "width": 249.83,
                    "height": 180.67,
                    "confidence": 0.27
                }],
                columns=['xcenter', 'ycenter', 'width', 'height', 'confidence']
            )
        ),
        (
            np.array(cv2.imread('Core/tests/Services/assets/non_target_image.jpg')),
            pd.DataFrame(
                columns=['xcenter', 'ycenter', 'width', 'height', 'confidence']
            )
        )
    ]
)

def test_yolo_v8_target_detection_predict_and_filter_by_confidence(yolo_model, image, expected_result):
    result = yolo_model.predict_and_filter_by_confidence(image, 0.1)
    assert_frame_equal(result, expected_result, check_dtype=False, atol=0.01)


