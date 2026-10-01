import time


# Drowsiness detector
class DrowsinessDetector:
    def __init__(
        self,
        threshold,
        warning_duration=1.0,
        drowsy_duration=2.0
    ):
        self.threshold = threshold
        self.warning_duration = warning_duration
        self.drowsy_duration = drowsy_duration

        self.closed_start_time = None
        self.state = "NORMAL"

    # Process both eyes
    def update(self, left_ear, right_ear):
        current_time = time.monotonic()

        if left_ear is None or right_ear is None:
            self.closed_start_time = None
            self.state = "NO_FACE"
            return self.state

        both_eyes_closed = (
            left_ear < self.threshold
            and right_ear < self.threshold
        )

        if both_eyes_closed:
            if self.closed_start_time is None:
                self.closed_start_time = current_time

            closed_duration = current_time - self.closed_start_time

            if closed_duration >= self.drowsy_duration:
                self.state = "DROWSY"
            elif closed_duration >= self.warning_duration:
                self.state = "WARNING"
            else:
                self.state = "CLOSING"

        else:
            self.closed_start_time = None
            self.state = "NORMAL"

        return self.state

    # Get closure duration
    def get_closed_duration(self):
        if self.closed_start_time is None:
            return 0.0

        return time.monotonic() - self.closed_start_time