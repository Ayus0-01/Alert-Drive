import cv2
import mediapipe as mp
import winsound

from drowsiness.detector import DrowsinessDetector
from drowsiness.thresholds import get_threshold
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
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    landmarker.close()
    exit()

# Detector setup
threshold = get_threshold()

detector = DrowsinessDetector(
    threshold=threshold,
    warning_duration=1.0,
    drowsy_duration=2.0
)

alert_active = False

# Extract eye points
def get_eye_points(face_landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = face_landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    return points

# Process frame
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None, []

    face_landmarks = result.face_landmarks[0]

    height, width, _ = frame.shape

    left_eye = get_eye_points(
        face_landmarks,
        LEFT_EYE_INDICES,
        width,
        height
    )

    right_eye = get_eye_points(
        face_landmarks,
        RIGHT_EYE_INDICES,
        width,
        height
    )

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    return left_ear, right_ear, left_eye + right_eye

# Live detection
while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    left_ear, right_ear, eye_points = process_frame(frame)

    state = detector.update(left_ear, right_ear)

    # Demo alert
    if state == "DROWSY" and not alert_active:
        print("DROWSY ALERT")
        winsound.Beep(1000, 500)
        alert_active = True

    elif state != "DROWSY":
        alert_active = False

    # Draw eye landmarks
    for point in eye_points:
        cv2.circle(
            frame,
            point,
            2,
            (0, 255, 0),
            -1
        )

    # Display EAR values
    if left_ear is not None and right_ear is not None:
        average_ear = (left_ear + right_ear) / 2.0

        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Average EAR: {average_ear:.3f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    else:
        cv2.putText(
            frame,
            "EAR: --",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Display detector state
    cv2.putText(
        frame,
        f"Threshold: {threshold:.3f}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"State: {state}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Live Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
landmarker.close()