import time
from enum import Enum
from typing import Tuple, List, NamedTuple
import matplotlib.pyplot as plt
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
    BOTTOM = 3


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
    color: Tuple[int, int, int] = attr.ib(default=(255, 0, 0))


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
    target_width: int
    target_height: int

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

        return cls(image, int(x0), int(y0), int(x1), int(y1), int(labels.width), int(labels.width))

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
            return round(abs(line.pt2.x - line.pt1.x) / CM_IN_PIXELS, 3)

        return round(abs(line.pt2.y - line.pt1.y) / CM_IN_PIXELS, 3)

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

            return abs(int(line.pt1.x * 1.1)), abs(line.pt1.y - 10)

        if line.side == CoordsSide.UP:
            return abs(line.pt1.x - 10), line.pt1.y // 2

        return abs(int(line.pt2.x - (line.pt2.x * 0.6))), int(line.pt1.y * 1.2)

    @property
    def line_to_left_side(self) -> Line:
        """
            This method is used to calculate the distance to the left side of the image.

        """
        return Line(
            Point(0, int(self.y0)),
            Point(int(self.x0), int(self.y0)),
            axis=0,
            side=CoordsSide.LEFT
        )

    @property
    def line_to_upper_side(self) -> Line:
        """
            This method is used to calculate the distance to the upper side of the image.
        """
        return Line(
            Point(int(self.x1), int(self.y0)),
            Point(int(self.x1), 0),
            axis=1,
            side=CoordsSide.UP
        )

    @property
    def line_from_top_side_to_center(self) -> Line:
        """
            This method is used to calculate the distance to the upper side of the image.
        """
        return Line(
            Point(int(self.x1) + 3, int(self.y0)),
            Point(int(self.x1) + 3, int(self.image.shape[0] / 2)),
            axis=1,
            side=CoordsSide.UP,
            color=(0, 0, 255)
        )

    @property
    def line_from_bottom_side_to_center(self) -> Line:
        """
            This method is used to calculate the distance to the upper side of the image.
        """
        return Line(
            Point(int(self.x0), int(self.y1)),
            Point(int(self.x0), int(self.image.shape[0] / 2)),
            axis=1,
            side=CoordsSide.UP,
            color=(0, 0, 255)
        )

    @property
    def line_from_right_side_to_center(self) -> Line:
        """
            This method is used to calculate the distance from the right target side to center.
        """
        return Line(
            Point(int(self.x1), int(self.y0)),
            Point(int(self.image.shape[1] / 2), int(self.y0)),
            axis=0,
            side=CoordsSide.RIGHT,
            color=(0, 0, 0)
        )

    @property
    def line_from_left_side_to_center(self) -> Line:
        """
            This method is used to calculate the distance from the right target side to center.
        """
        return (Line(
            Point(int(self.x0), int(self.y0)),
            Point(int(self.image.shape[1] / 2), int(self.y0)),
            axis=0,
            side=CoordsSide.LEFT,
            color=(0, 0, 0)
        ))

    @property
    def line_from_x_center_to_center_image(self) -> Line:
        """
            This method is used to calculate the distance from the right target side to center.
        """
        return (Line(
            Point(int(self.x0 + self.target_width / 2), int(self.y0)),
            Point(int(self.image.shape[1] / 2), int(self.y0)),
            axis=0,
            side=CoordsSide.LEFT,
            color=(0, 0, 0)
        ))

    @property
    def line_to_right_side(self) -> Line:
        """
            This method is used to calculate the distance to the right side of the image.
        """

        return Line(
            Point(int(self.x1), int(self.x1)),
            Point(int(self.image.shape[1]), int(self.x1)),
            axis=0,
            side=CoordsSide.RIGHT
        )

    @property
    def line_to_bottom_side(self) -> Line:
        """
            This method is used to calculate the distance to the bottom side of the image.
        """
        return Line(
            Point(int(self.x1), int(self.y1)),
            Point(int(self.x1), int(self.image.shape[0])),
            axis=1,
            side=CoordsSide.BOTTOM
        )

    @property
    def line_to_max_width(self) -> Line:
        """
            This method is used to calculate the distance to the max width of the image.
        """
        return Line(
            Point(0, int(self.image.shape[0] / 2)),
            Point(int(self.image.shape[1]), int(self.image.shape[0] / 2)),
            axis=0,
            color=(255, 0, 0)
        )

    @property
    def line_to_max_height(self) -> Line:
        """
            This method is used to calculate the distance to the max height of the image.
        """
        return Line(
            Point(int(self.image.shape[1] / 2), 0),
            Point(int(self.image.shape[1] / 2), int(self.image.shape[0])),
            axis=1,
            color=(255, 0, 0)
        )

    def get_all_lines(self) -> List[Line]:
        """
            This method is used to get all lines.
        """
        return [
            self.line_to_left_side,
            self.line_to_upper_side,
            self.line_to_right_side,
            self.line_to_bottom_side,
            self.line_from_top_side_to_center,
            self.line_from_bottom_side_to_center,
            self.line_from_right_side_to_center,
            self.line_from_left_side_to_center,
            self.line_from_x_center_to_center_image,
            self.line_to_max_width,
            self.line_to_max_height,
        ]

    def get_all_distances(self) -> pd.Series:
        """
            This method is used to get all distances.
        """
        return pd.Series(
            [self.get_distance_in_cm(line) for line in self.get_all_lines()],
            index=[
                'left', 'top', 'right', 'bottom', 'from_top_side_to_center', 'from_bottom_side_to_center',
                "from_right_side_to_center", "from_left_side_to_center",
                "from_x_center_to_center_image", 'width', 'height'
            ]
        )

    def draw_lines_into_image(
            self,
            wait_time_window: int,
            lines: List[Line] = None,
            use_cv2=True
    ) -> np.array:
        """
        This method is used to draw lines into the image.

        Parameters
        ----------
        wait_time_window: int
            This parameter contains the time to wait to close the window.
        lines: List[Line]
            This parameter contains the lines to be drawn into the image.
        use_cv2: bool
            This parameter contains a boolean to use cv2 or use matplotlib to show images.
        """

        if lines is None:
            lines = self.get_all_lines()

        cv2.startWindowThread()
        cv2.rectangle(
            self.image,
            (self.x0, self.y0),
            (self.x1, self.y1),
            (36, 255, 12),
            3
        )

        for line in lines:
            cv2.line(self.image, line.pt1, line.pt2, line.color, 1)
            cv2.putText(
                self.image,
                f'{self.get_distance_in_cm(line)} cm',
                self.calculate_coords_text_cm(line),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                line.color,
                1,
                cv2.LINE_AA
            )

        if not use_cv2:
            manager = plt.get_current_fig_manager()
            manager.window.wm_geometry("+100+100")
            plt.imshow(self.image)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.show(block=False)
            plt.pause(wait_time_window / 1000)
            plt.close()
            return self.image

        cv2.imshow('lines', self.image)
        cv2.waitKey(wait_time_window)
        cv2.destroyAllWindows()

        return self.image
