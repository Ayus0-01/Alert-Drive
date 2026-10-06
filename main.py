import cv2
import mediapipe as mp

from drowsiness.detector import DrowsinessDetector
from drowsiness.thresholds import get_threshold
from hardware.controller import AlertController
from hardware.demo_alert import DemoAlert
from utils.ear import calculate_ear


# Eye landmark indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="assets/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# Camera setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


# Detection setup
threshold = get_threshold()

detector = DrowsinessDetector(
    threshold=threshold,
    warning_duration=1.0,
    drowsy_duration=2.0
)

alert_controller = AlertController()
demo_alert = DemoAlert()


def get_eye_points(face_landmarks, indices, width, height):
    normalized_points = []
    pixel_points = []

    for index in indices:
        landmark = face_landmarks[index]

        # Keep normalized precision for EAR
        normalized_points.append(
            (landmark.x, landmark.y)
        )

        # Convert only for drawing
        pixel_points.append(
            (
                int(landmark.x * width),
                int(landmark.y * height)
            )
        )

    return normalized_points, pixel_points


def process_frame(frame):
    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None, None, None

    face_landmarks = result.face_landmarks[0]

    left_eye, left_eye_pixels = get_eye_points(
        face_landmarks,
        LEFT_EYE_INDICES,
        width,
        height
    )

    right_eye, right_eye_pixels = get_eye_points(
        face_landmarks,
        RIGHT_EYE_INDICES,
        width,
        height
    )

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    average_ear = (left_ear + right_ear) / 2.0

    return (
        left_ear,
        right_ear,
        average_ear,
        left_eye_pixels,
        right_eye_pixels
    )


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    result = process_frame(frame)

    if result[0] is None:
        left_ear = None
        right_ear = None
        average_ear = None
        left_eye_pixels = []
        right_eye_pixels = []
    else:
        (
            left_ear,
            right_ear,
            average_ear,
            left_eye_pixels,
            right_eye_pixels
        ) = result

    state = detector.update(
        left_ear,
        right_ear
    )

    alert_controller.update(state)
    demo_alert.update(state)

    # Draw landmarks
    for point in left_eye_pixels:
        cv2.circle(
            frame,
            point,
            2,
            (0, 255, 0),
            -1
        )

    for point in right_eye_pixels:
        cv2.circle(
            frame,
            point,
            2,
            (0, 255, 0),
            -1
        )

    # Display values
    if average_ear is not None:
        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.3f}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Average EAR: {average_ear:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"Threshold: {threshold:.3f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"State: {state}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()