import cv2
import mediapipe as mp
import time

from utils.ear import calculate_ear


LEFT_EYE_INDICES = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE_INDICES = [
    362,
    385,
    387,
    263,
    373,
    380
]


def get_eye_points(face_landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = face_landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    return points


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

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera failed")
    exit()


# --------------------------------------------------
# Calibration state
# --------------------------------------------------

samples = []

start_time = time.time()

CALIBRATION_DURATION = 5


print()
print("=" * 50)
print("EAR CALIBRATION")
print("=" * 50)
print()
print("For the next 5 seconds:")
print()
print("  Keep your eyes normally OPEN.")
print("  Look naturally at the camera.")
print("  Blink normally.")
print()
print("Do NOT deliberately close your eyes.")
print()
print("Starting...")
print()


while True:

    success, frame = camera.read()

    if not success:
        print("Could not receive frame")
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    elapsed = time.time() - start_time
    remaining = CALIBRATION_DURATION - elapsed

    if result.face_landmarks:

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

        ear = (left_ear + right_ear) / 2.0

        samples.append(ear)

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Calibration: {max(0, remaining):.1f}s",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "EAR Calibration",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if elapsed >= CALIBRATION_DURATION:
        break


camera.release()
cv2.destroyAllWindows()
landmarker.close()


# --------------------------------------------------
# Results
# --------------------------------------------------

if not samples:

    print()
    print("ERROR: No EAR samples were collected.")
    print("Make sure your face was visible.")
    exit()


average_ear = sum(samples) / len(samples)
minimum_ear = min(samples)
maximum_ear = max(samples)


print()
print("=" * 50)
print("CALIBRATION RESULTS")
print("=" * 50)

print(f"Samples collected : {len(samples)}")
print(f"Average EAR       : {average_ear:.4f}")
print(f"Minimum EAR       : {minimum_ear:.4f}")
print(f"Maximum EAR       : {maximum_ear:.4f}")

print("=" * 50)