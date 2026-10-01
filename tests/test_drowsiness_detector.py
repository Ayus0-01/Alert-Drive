import time

from drowsiness.detector import DrowsinessDetector


# Test detector states
detector = DrowsinessDetector(
    threshold=0.1600,
    warning_duration=1.0,
    drowsy_duration=2.0
)

print("Initial state:", detector.state)

print("Both eyes open:", detector.update(0.20, 0.20))

print("Left closed, right open:", detector.update(0.10, 0.20))

time.sleep(1.2)

print(
    "Left closed, right open after 1.2 seconds:",
    detector.update(0.10, 0.20)
)

print("Both eyes closed:", detector.update(0.10, 0.10))

time.sleep(1.2)

print(
    "Both eyes closed after 1.2 seconds:",
    detector.update(0.10, 0.10)
)

time.sleep(1.0)

print(
    "Both eyes closed after 2.2 seconds:",
    detector.update(0.10, 0.10)
)

print("Both eyes open:", detector.update(0.20, 0.20))

print("No face:", detector.update(None, None))