import math
from typing import Sequence, Tuple


Point = Tuple[int, int]


def euclidean_distance(point_a: Point, point_b: Point) -> float:
    """
    Calculate the Euclidean distance between two 2D points.
    """

    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1]
    )


def calculate_ear(eye_points: Sequence[Point]) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR).

    Expected point order:

        p1 ---- p4
         \      /
          p2  p3
          |    |
          p6  p5

    EAR = ((p2-p6) + (p3-p5)) / (2 * (p1-p4))

    Parameters
    ----------
    eye_points:
        Six 2D eye landmark coordinates in the required order.

    Returns
    -------
    float
        Eye Aspect Ratio.

    Raises
    ------
    ValueError
        If the number of supplied landmarks is not six or
        if the horizontal eye distance is zero.
    """

    if len(eye_points) != 6:
        raise ValueError(
            f"EAR calculation requires exactly 6 points, "
            f"received {len(eye_points)}."
        )

    p1, p2, p3, p4, p5, p6 = eye_points

    horizontal_distance = euclidean_distance(p1, p4)

    if horizontal_distance == 0:
        raise ValueError(
            "Cannot calculate EAR because horizontal eye distance is zero."
        )

    vertical_distance_1 = euclidean_distance(p2, p6)
    vertical_distance_2 = euclidean_distance(p3, p5)

    ear = (
        vertical_distance_1 + vertical_distance_2
    ) / (2.0 * horizontal_distance)

    return ear