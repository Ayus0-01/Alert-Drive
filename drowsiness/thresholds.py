import json


# Calibration file
CALIBRATION_FILE = "config/calibration.json"


# Load calibration
def load_calibration():
    with open(CALIBRATION_FILE, "r") as file:
        return json.load(file)


# Calculate threshold
def calculate_threshold(calibration):
    open_p90 = calibration["open_eyes"]["p90"]
    closed_p90 = calibration["closed_eyes"]["p90"]

    return (open_p90 + closed_p90) / 2.0


# Get threshold
def get_threshold():
    calibration = load_calibration()
    return calculate_threshold(calibration)


if __name__ == "__main__":
    calibration = load_calibration()
    threshold = calculate_threshold(calibration)

    print()
    print("=" * 50)
    print("EAR THRESHOLD")
    print("=" * 50)
    print(f"Open-eye P90   : {calibration['open_eyes']['p90']:.4f}")
    print(f"Closed-eye P90 : {calibration['closed_eyes']['p90']:.4f}")
    print(f"Threshold      : {threshold:.4f}")
    print("=" * 50)