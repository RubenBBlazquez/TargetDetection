import logging
from enum import Enum
from typing import Tuple, List, NamedTuple

import attr
import cv2
import numpy as np
import pandas as pd
from attr import define

CM_IN_PIXELS = 37.7952755906


class CoordsSide(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Point(NamedTuple):
    """
        This class is used to represent a point in a line.

        Attributes:
        -----------
        x: int
            This attribute contains the x coordinate of the point.
        y: int
            This attribute contains the y coordinate of the point.
    """
    x: int
    y: int


@define(auto_attribs=True)
class Line:
    """
        This class is used to represent a line.

        Attributes:
        -----------
        pt1: Point
            This attribute contains the first point of the line.
        pt2: Point
            This attribute contains the second point of the line.
        axis: int
            This attribute contains the axis of the line.
        side: CoordsSide
            This attribute contains the side of the line.
    """
    pt1: Point
    pt2: Point
    axis: int = attr.ib(default=0)
    side: CoordsSide = attr.ib(default=CoordsSide.LEFT)


@define(auto_attribs=True)
class DistanceCalculations:
    """
        This class is used to calculate distance based on an image and labels.

        Attributes:
        -----------
        image: np.array
            This attribute contains the image.
        x0: int
            This attribute contains the x0 coordinate of the labels.
        y0: int
            This attribute contains the y0 coordinate of the labels.
        x1: int
            This attribute contains the x1 coordinate of the labels.
        y1: int
            This attribute contains the y1 coordinate of the labels.
    """
    image: np.array
    x0: int
    y0: int
    x1: int
    y1: int

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

        return cls(image, int(x0), int(y0), int(x1), int(y1))

    def get_distance_in_cm(self, line: Line) -> float:
        """
        This method is used to calculate the distance in cm.

        Parameters
        ----------
        axis:
          define if we want to calculate distance in x-axis or y-axis
        line: Line
            define the line to calculate the distance
        """

        if line.axis == 0:
            return abs(line.pt2.x - line.pt1.x) / CM_IN_PIXELS

        return abs(line.pt2.y - line.pt1.y) / CM_IN_PIXELS

    def distance_to_left(self) -> Line:
        """
            This property is used to calculate the distance to the left side of the image.

            Returns:
            --------
            float
                This method returns the distance to the left side of the image in meters.
        """
        return Line(
            Point(0, int(self.y0)),
            Point(int(self.x0), int(self.y0)),
            axis=0,
            side=CoordsSide.LEFT
        )

    def distance_to_up(self) -> Line:
        """
            This property is used to calculate the distance to the upper side of the image.

            Returns:
            --------
            float
                This method returns the distance to the left side of the image in meters.
        """
        return Line(
            Point(int(self.x1), int(self.y0)),
            Point(int(self.x1), 0),
            axis=1,
            side=CoordsSide.UP
        )

    def calculate_coords_text_cm(self, line: Line) -> Tuple[int, int]:
        """
        This method is used to calculate the coordinates of the text in cm.

        Parameters
        ----------
        line: Line
            This parameter contains the line to calculate the coordinates of the text in cm.
        """
        if line.axis == 0:
            if line.side == CoordsSide.LEFT:
                return line.pt2.x // 2, abs(line.pt2.y - 10)

            return line.pt1.x // 2, abs(line.pt1.y - 10)

        if line.side == CoordsSide.UP:
            return abs(line.pt1.x - 10), line.pt1.y // 2

        return abs(line.pt1.x - 10), line.pt1.y // 2

    def draw_lines_into_image(self, lines: List[Line]) -> None:
        """
        This method is used to draw lines into the image.

        Parameters
        ----------
        lines: List[Line]
            This parameter contains the lines to be drawn into the image.
        """
        cv2.rectangle(
            self.image,
            (self.x0, self.y0),
            (self.x1, self.y1),
            (36, 255, 12),
            1
        )

        for line in lines:
            logging.error(f"line: {({line.pt2.x // 2}, {line.pt2.y - 10})}")
            cv2.line(self.image, line.pt1, line.pt2, (255, 0, 0), 1)
            cv2.putText(
                self.image,
                f'{round(self.get_distance_in_cm(line), 3)} cm',
                self.calculate_coords_text_cm(line),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )

        cv2.imshow('lines', self.image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        return self.image
