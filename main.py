import cv2
import mediapipe as mp

from utils.ear import calculate_ear


# --------------------------------------------------
# MediaPipe eye landmark indices
# --------------------------------------------------

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


# --------------------------------------------------
# Eye landmark extraction
# --------------------------------------------------

def get_eye_points(face_landmarks, eye_indices, width, height):
    """
    Convert MediaPipe normalized landmarks into
    pixel coordinates.
    """

    points = []

    for index in eye_indices:

        landmark = face_landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    return points


# --------------------------------------------------
# Create Face Landmarker
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="assets/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=2
)


landmarker = FaceLandmarker.create_from_options(options)

# Open camera

camera = cv2.VideoCapture(0)


if not camera.isOpened():
    print("Camera failed")
    exit()

# Main processing loop

while True:

    success, frame = camera.read()

    if not success:
        print("Could not receive frame")
        break

    # Convert OpenCV BGR image to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert image into MediaPipe format
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect facial landmarks
    result = landmarker.detect(mp_image)

    # Process detected face

    if result.face_landmarks:

        # Primary face
        face_landmarks = result.face_landmarks[0]

        height, width, _ = frame.shape

        # Extract eye landmarks

        left_eye_points = get_eye_points(
            face_landmarks,
            LEFT_EYE_INDICES,
            width,
            height
        )

        right_eye_points = get_eye_points(
            face_landmarks,
            RIGHT_EYE_INDICES,
            width,
            height
        )

        # Calculate EAR

        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)

        ear = (left_ear + right_ear) / 2.0

        # Display EAR

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Draw eye landmarks

        for point in left_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 0),
                -1
            )


        for point in right_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 0),
                -1
            )

    # Display frame

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )

    # Quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup

camera.release()
cv2.destroyAllWindows()

landmarker.close()