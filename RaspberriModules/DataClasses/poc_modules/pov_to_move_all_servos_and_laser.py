from ServoModule import ServoMovement
import time
import random
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

def servo2():
    GPIO.setup(15, GPIO.OUT)
    GPIO.output(15, GPIO.HIGH)
    time.sleep(0.1)
    servo2 = ServoMovement(13, 9.5)
    servo2.default_move()
    servo2.stop()
    time.sleep(0.5)
    servo2 = ServoMovement(13, 12.4)
    servo2.default_move()
    servo2.stop()
    time.sleep(0.5)
    GPIO.setup(15, GPIO.OUT)
    GPIO.output(15, GPIO.LOW)
    time.sleep(0.1)

while True:
    position = 2
    servo = ServoMovement(11, position)
    servo2()
    time.sleep(1)
    print(servo)

    rn = random.randint(1, 4)
    for index, n in enumerate([2, 10]):
        print(rn, index, n)
        servo = ServoMovement(11, n)
        servo.move_to(n)

        time.sleep(5)

