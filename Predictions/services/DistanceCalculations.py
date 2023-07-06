import numpy as np
import pandas as pd
from attr import define

CM_IN_PIXELS = 37.8


@define(auto_attribs=True)
class DistanceCalculations:
    """
        This class is used to calculate distance based on an image and labels.

        Attributes:
        -----------
        distance: float
            This attribute contains the distance in meters.
        angle: float
            This attribute contains the angle in degrees.
        servo_position: int
            This attribute contains the servo position.
        date: datetime
            This attribute contains the date when the calculation was made.
    """
    image: np.array
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def create_from(cls, image: np.array, labels: pd.Series):
        """
            This method is used to create a DistanceCalculations instance from an image and labels.

            Parameters:
            -----------
            image: np.array
                This parameter contains the image.
            labels: pd.Series
                This parameter contains the labels.
        """
        x0 = labels.xcenter - labels.width / 2
        y0 = labels.ycenter - labels.height / 2
        x1 = labels.xcenter + labels.width / 2
        y1 = labels.ycenter + labels.height / 2

        return cls(image, x0, y0, x1, y1)

    @property
    def distance_to_left(self) -> float:
        """
            This property is used to calculate the distance to the left side of the image.

            Returns:
            --------
            float
                This method returns the distance to the left side of the image in meters.
        """
        return self.x0 / CM_IN_PIXELS

    def draw_lines_into_image(self, lines: list) -> None:
        cv2.dra
