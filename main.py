
import cv2
import mediapipe as mp

from drowsiness.detector import DrowsinessDetector
from drowsiness.thresholds import get_threshold
from hardware.controller import AlertController
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


# Extract eye landmarks
def get_eye_points(face_landmarks, indices, width, height):
    ear_points = []
    pixel_points = []

    for index in indices:
        landmark = face_landmarks[index]

        x = landmark.x * width
        y = landmark.y * height

        ear_points.append((x, y))
        pixel_points.append((int(x), int(y)))

    return ear_points, pixel_points


# Process camera frame
def process_frame(frame, landmarker):
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
        return None, None, None, [], []

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


def main():
    landmarker = None
    cap = None

    try:
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )

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

        print("Driver Drowsiness Detection Started")
        print(f"EAR Threshold: {threshold:.4f}")
        print("Press Q to exit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read camera frame.")
                break

            (
                left_ear,
                right_ear,
                average_ear,
                left_eye_pixels,
                right_eye_pixels
            ) = process_frame(frame, landmarker)

            # Update detection state
            state = detector.update(
                left_ear,
                right_ear
            )

            # Update alert controller
            alert_controller.update(state)

            # Draw eye landmarks
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

            # Display EAR values
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
            else:
                cv2.putText(
                    frame,
                    "Face not detected",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # Display threshold and state
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

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
