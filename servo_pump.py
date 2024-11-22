import RPi.GPIO as GPIO
from time import sleep
from gpiozero import AngularServo

RELAY_1_GPIO = 26
SERVO_1_GPIO = 11

GPIO.setmode(GPIO.BCM) # GPIO Numbers instead of board numbers 
GPIO.setup(RELAY_1_GPIO, GPIO.OUT) # GPIO Assign mode
GPIO.output(RELAY_1_GPIO, GPIO.LOW)

servo = AngularServo(SERVO_1_GPIO, min_angle =0, max_angle = 270, min_pulse_width =0.5/1000, max_pulse_width=2.5/1000)
servo.detach()

def set_angle(angle):
    servo.angle=angle
    sleep(1)

def inc_angle(angle):
    if(servo.angle+angle<=270 and servo.angle+angle>=0):
        servo.angle+=angle
        sleep(0.05)    

def start_shooting():
    GPIO.output(RELAY_1_GPIO, GPIO.HIGH)
    sleep(0.5)

def stop_shooting():
    GPIO.output(RELAY_1_GPIO, GPIO.LOW)
    servo.detach()
    sleep(0.5)

try:
    while True:
        angle = int(input("enter angle: "))
        set_angle(angle)
        servo.detach()
        GPIO.output(RELAY_1_GPIO, GPIO.HIGH)
        sleep(5)
        GPIO.output(RELAY_1_GPIO, GPIO.LOW)
except KeyboardInterrupt:
    print("stopped")