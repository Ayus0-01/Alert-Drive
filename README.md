# AI-Based Driver Drowsiness Detection and Alert System

## 1. Project Overview

The AI-Based Driver Drowsiness Detection and Alert System is a computer-vision-based system designed to monitor a driver's eye activity and identify prolonged eye closure.

The system uses a camera and MediaPipe Face Landmarker to detect facial landmarks. It calculates the Eye Aspect Ratio (EAR) to estimate eye closure and uses a time-based detection mechanism to classify the driver's state.

The project is being developed with a modular architecture to support future hardware integration for physical alerts.

## 2. Current Features

- Real-time camera input using OpenCV.
- Facial landmark detection using MediaPipe.
- Left-eye and right-eye landmark extraction.
- Eye Aspect Ratio (EAR) calculation using NumPy.
- Individual eye measurements and average EAR calculation.
- Controlled open-eye and closed-eye calibration.
- Calibration data stored in JSON format.
- Configurable EAR threshold calculated from calibration data.
- Time-based drowsiness state detection.
- Separate alert controller for future hardware integration.
- Unit testing for the drowsiness detection logic.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| OpenCV | Camera input and image processing |
| MediaPipe | Facial landmark detection |
| NumPy | Numerical calculations |
| JSON | Calibration data storage |

## 4. Project Structure

```text
Driver_Drowsiness_Detection/
│
├── assets/
│   └── face_landmarker.task
│
├── config/
│   └── calibration.json
│
├── drowsiness/
│   ├── __init__.py
│   ├── detector.py
│   └── thresholds.py
│
├── hardware/
│   ├── __init__.py
│   └── controller.py
│
├── tests/
│   ├── live_drowsiness_test.py
│   └── test_drowsiness_detector.py
│
├── utils/
│   ├── __init__.py
│   ├── controlled_calibration.py
│   ├── ear.py
│   └── ear_calibration.py
│
├── main.py
├── requirements.txt
└── README.md
```

## 5. System Architecture

The current software pipeline follows this sequence:

1. Camera captures the driver's face.
2. OpenCV reads and processes the camera frame.
3. MediaPipe identifies facial landmarks.
4. Eye landmarks are extracted for both eyes.
5. EAR is calculated separately for the left and right eyes.
6. The detector evaluates both EAR values against the calibrated threshold.
7. The detector determines the current drowsiness state.
8. The alert controller receives the state for future hardware output.

## 6. Drowsiness Detection Logic

The system currently uses six facial landmarks for each eye to calculate the Eye Aspect Ratio.

The EAR formula is:

EAR = (||P2 - P6|| + ||P3 - P5||) / (2 × ||P1 - P4||)

The detector evaluates both eyes independently.

The current detection states are:

| State | Description |
|---|---|
| NORMAL | Both eyes are above the closure threshold |
| CLOSING | Both eyes are below the threshold for less than the warning duration |
| WARNING | Both eyes remain below the threshold for at least 1 second |
| DROWSY | Both eyes remain below the threshold for at least 2 seconds |
| NO_FACE | No face is detected |

A single-eye closure does not trigger the current drowsiness sequence.

## 7. Calibration

The system uses controlled calibration to collect EAR samples under two conditions:

- Open-eye phase: The user keeps their eyes naturally open and blinks normally.
- Closed-eye phase: The user keeps their eyes closed.

The collected measurements include:

- Mean
- Median
- Minimum
- Maximum
- Standard deviation
- 10th percentile
- 90th percentile

The calibration results are saved in:

`config/calibration.json`

The current threshold calculation uses the midpoint between the open-eye P90 and closed-eye P90 values.

## 8. Installation and Setup

### Clone the repository

```bash
git clone <repository-url>
cd Driver_Drowsiness_Detection
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

Windows CMD:

```cmd
venv\Scripts\activate
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Ensure the MediaPipe model file exists at:

`assets/face_landmarker.task`

## 9. Running the Project

### Run controlled calibration

From the project root:

```bash
python -m utils.controlled_calibration
```

### Run the main application

```bash
python main.py
```

Press `Q` to exit the camera window.

### Run detector tests

```bash
python tests/test_drowsiness_detector.py
```

### Run live detection test

```bash
python tests/live_drowsiness_test.py
```

## 10. Current Development Status

The software currently implements camera-based eye monitoring, facial landmark extraction, EAR calculation, calibration, time-based drowsiness classification, and a modular alert controller.

The previous Windows sound-based demonstration alert is being removed in preparation for physical hardware integration.

Hardware communication and physical alert activation are not yet implemented.

## 11. Planned Improvements

### Phase 1 — Detection Improvements
- Improve eye landmark visualization and tracking.
- Standardize coordinate calculations across the application and calibration tools.
- Evaluate EAR stability under different lighting and head positions.
- Introduce appropriate temporal smoothing and noise handling.
- Improve calibration reliability.

### Phase 2 — Hardware Integration
- Finalize the microcontroller and hardware components.
- Establish communication between Python and the microcontroller.
- Integrate LED, buzzer, and vibration motor outputs.
- Map detection states to physical alert behavior.
- Implement hardware communication error handling.

### Phase 3 — System Optimization
- Improve real-time processing performance.
- Evaluate detection reliability across different users.
- Test different lighting conditions and camera positions.
- Improve recovery and error-handling behavior.

### Phase 4 — Product Development
- Integrate software and hardware into a compact system.
- Develop a stable startup and shutdown process.
- Conduct real-world testing.
- Evaluate system reliability and false-alert rates.

## 12. Project Objective

The ultimate objective is to develop a compact, reliable, and affordable driver drowsiness monitoring system that detects prolonged eye closure and provides immediate physical alerts to help reduce risks associated with driver fatigue.