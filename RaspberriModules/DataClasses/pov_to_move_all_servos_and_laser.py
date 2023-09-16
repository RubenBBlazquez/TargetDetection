from RaspberriModules.DataClasses.ServoModule import ServoMovement
import time
import random
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

def servo2():
    GPIO.setup(15, GPIO.OUT)
    GPIO.output(15, GPIO.HIGH)
    time.sleep(0.1)
    servo2 = ServoMovement(13, 9.5)
    servo2.stop()
    time.sleep(0.5)
    servo2 = ServoMovement(13, 12.4)
    servo2.stop()
    time.sleep(0.5)
    GPIO.setup(15, GPIO.OUT)
    GPIO.output(15, GPIO.LOW)
    time.sleep(0.1)
    
while True:
    servo2()
    position = 2
    servo = ServoMovement(11, position)
    servo.stop()
    time.sleep(1)
    print(servo)

    rn = random.randint(1, 4)
    for index, n in enumerate([6, 9 , 15]):
        print(rn, index, n)
        servo = ServoMovement(11, n)
        servo.stop()

        if rn == index:
            servo2()

        time.sleep(1)
    
        
    rn = random.randint(1, 3)
    for index, n in enumerate([6, 9][::-1]):
        print(rn, index, n)
        servo = ServoMovement(11, n)
        servo.stop()

        if rn == index:
            servo2()

        time.sleep(1)

