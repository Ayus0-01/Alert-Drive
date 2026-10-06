import numpy as np


def euclidean_distance(point_a, point_b):
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)

    return float(np.linalg.norm(point_a - point_b))


def calculate_ear(eye_points):
    if len(eye_points) != 6:
        raise ValueError(
            f"EAR calculation requires exactly 6 points, received {len(eye_points)}."
        )

    p1, p2, p3, p4, p5, p6 = eye_points

    horizontal_distance = euclidean_distance(p1, p4)

    if horizontal_distance == 0:
        raise ValueError(
            "Cannot calculate EAR because horizontal eye distance is zero."
        )

    vertical_distance_1 = euclidean_distance(p2, p6)
    vertical_distance_2 = euclidean_distance(p3, p5)

    return (
        vertical_distance_1 + vertical_distance_2
    ) / (2.0 * horizontal_distance)