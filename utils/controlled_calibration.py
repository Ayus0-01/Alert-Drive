import cv2
import mediapipe as mp
import time
import statistics
import json
import os

from utils.ear import calculate_ear


# Eye landmark indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

OPEN_DURATION = 5
CLOSED_DURATION = 5


# Extract eye points
def get_eye_points(face_landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = face_landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    return points


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


# Calculate current EAR
def get_current_ear(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

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

    return (left_ear + right_ear) / 2.0


# Run calibration phase
def run_phase(title, instruction, duration):
    samples = []
    start_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        elapsed = time.time() - start_time
        remaining = duration - elapsed

        ear = get_current_ear(frame)

        cv2.putText(
            frame,
            title,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            instruction,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Time remaining: {max(0, remaining):.1f}s",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if ear is not None:
            samples.append(ear)

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("EAR Controlled Calibration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return samples, False

        if elapsed >= duration:
            break

    return samples, True


# Calibration statistics
def calculate_statistics(samples):
    if not samples:
        return None

    ordered = sorted(samples)

    return {
        "samples": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "std_dev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p10": ordered[int((len(ordered) - 1) * 0.10)],
        "p90": ordered[int((len(ordered) - 1) * 0.90)]
    }

# Save calibration data
def save_calibration(open_stats, closed_stats):
    os.makedirs("config", exist_ok=True)

    calibration = {
        "open_eyes": open_stats,
        "closed_eyes": closed_stats
    }

    with open("config/calibration.json", "w") as file:
        json.dump(calibration, file, indent=4)

    print()
    print("Calibration saved to config/calibration.json")


# Start calibration
print()
print("=" * 60)
print("CONTROLLED EAR CALIBRATION")
print("=" * 60)
print()
print("Phase 1: Keep your eyes naturally OPEN.")
print("Phase 2: Keep your eyes CLOSED.")
print("Keep your head reasonably still.")
print("Press Q to cancel.")
print()
print("Starting in 3 seconds...")

time.sleep(3)


# Open-eye phase
open_samples, completed = run_phase(
    "PHASE 1 — OPEN EYES",
    "Keep eyes naturally OPEN and blink normally.",
    OPEN_DURATION
)

if not completed:
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Calibration cancelled.")
    exit()


print()
print("Open-eye phase complete.")
print()


# Transition
for countdown in [3, 2, 1]:
    success, frame = camera.read()

    if not success:
        break

    cv2.putText(
        frame,
        f"NEXT: CLOSE YOUR EYES — {countdown}",
        (50, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("EAR Controlled Calibration", frame)
    cv2.waitKey(1000)


# Closed-eye phase
closed_samples, completed = run_phase(
    "PHASE 2 — CLOSED EYES",
    "Keep your eyes CLOSED for the entire phase.",
    CLOSED_DURATION
)


# Cleanup
camera.release()
cv2.destroyAllWindows()
landmarker.close()

if not completed:
    print("Calibration cancelled.")
    exit()


# Generate results
open_stats = calculate_statistics(open_samples)
closed_stats = calculate_statistics(closed_samples)


# Display results
print()
print("=" * 60)
print("CONTROLLED CALIBRATION RESULTS")
print("=" * 60)

print()
print("OPEN EYES")
print("-" * 60)

if open_stats:
    print(f"Samples       : {open_stats['samples']}")
    print(f"Mean EAR      : {open_stats['mean']:.4f}")
    print(f"Median EAR    : {open_stats['median']:.4f}")
    print(f"Minimum EAR   : {open_stats['minimum']:.4f}")
    print(f"Maximum EAR   : {open_stats['maximum']:.4f}")
    print(f"Std deviation : {open_stats['std_dev']:.4f}")
    print(f"P10           : {open_stats['p10']:.4f}")
    print(f"P90           : {open_stats['p90']:.4f}")
else:
    print("No open-eye samples collected.")


print()
print("CLOSED EYES")
print("-" * 60)

if closed_stats:
    print(f"Samples       : {closed_stats['samples']}")
    print(f"Mean EAR      : {closed_stats['mean']:.4f}")
    print(f"Median EAR    : {closed_stats['median']:.4f}")
    print(f"Minimum EAR   : {closed_stats['minimum']:.4f}")
    print(f"Maximum EAR   : {closed_stats['maximum']:.4f}")
    print(f"Std deviation : {closed_stats['std_dev']:.4f}")
    print(f"P10           : {closed_stats['p10']:.4f}")
    print(f"P90           : {closed_stats['p90']:.4f}")
else:
    print("No closed-eye samples collected.")


print()
print("=" * 60)
print("CALIBRATION COMPLETE")
print("=" * 60)

if open_stats and closed_stats:
    save_calibration(open_stats, closed_stats)