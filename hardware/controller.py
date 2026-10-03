# Alert controller
class AlertController:
    def __init__(self):
        self.current_state = "NORMAL"

    # Update alert state
    def update(self, state):
        if state == self.current_state:
            return

        self.current_state = state
        self._handle_state(state)

    # Handle alert state
    def _handle_state(self, state):
        if state == "NORMAL":
            self._normal()

        elif state == "CLOSING":
            self._closing()

        elif state == "WARNING":
            self._warning()

        elif state == "DROWSY":
            self._drowsy()

        elif state == "NO_FACE":
            self._no_face()

    # Normal state
    def _normal(self):
        print("ALERT: NORMAL")

    # Closing state
    def _closing(self):
        print("ALERT: CLOSING")

    # Warning state
    def _warning(self):
        print("ALERT: WARNING")

    # Drowsy state
    def _drowsy(self):
        print("ALERT: DROWSY")

    # No-face state
    def _no_face(self):
        print("ALERT: NO_FACE")