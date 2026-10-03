import time

from hardware.demo_alert import DemoAlert


# Test demo alert
alert = DemoAlert()

alert.update("NORMAL")

alert.update("WARNING")

alert.update("DROWSY")

time.sleep(1)

alert.update("DROWSY")

alert.update("NORMAL")

alert.update("DROWSY")