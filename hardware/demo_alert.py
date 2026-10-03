import winsound


# Demo alert output
class DemoAlert:
    def __init__(self):
        self.active = False

    # Update alert
    def update(self, state):
        if state == "DROWSY" and not self.active:
            print("DROWSY ALERT")
            winsound.Beep(1000, 500)
            self.active = True

        elif state != "DROWSY":
            self.active = False