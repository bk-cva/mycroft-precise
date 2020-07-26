import RPi.GPIO as GPIO    # Import Raspberry Pi GPIO library
from time import sleep     # Import the sleep function from the time module
from enum import Enum


class DeviceGPIO(Enum):
    Aircon = 8
    LeftWindow = 10
    LeftDoor = 12
    Radio = 36
    RightWindow = 38
    RightDoor = 40

devices_car = DeviceGPIO
GPIO.setwarnings(False)    # Ignore warning for now
GPIO.setmode(GPIO.BOARD)   # Use physical pin numbering

for d in devices_car:
    print(d.value)
    GPIO.setup(d.value, GPIO.OUT)

while True: # Run forever
    for d in devices_car:
    	GPIO.output([8, 10, 12], 0) # Turn on
    	sleep(1)                  # Sleep for 1 second
    	GPIO.output(d.value, GPIO.LOW)  # Turn off
    	sleep(0.1)
