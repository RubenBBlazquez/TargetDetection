from time import sleep
from attrs import define, field
import RPi.GPIO as GPIO


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class ServoGPIOModule(metaclass=SingletonMeta):
    def __init__(self, gpin: int) -> None:
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(gpin, GPIO.OUT)
        self.servo = GPIO.PWM(gpin, 50)
        self.servo.start(0)


@define(auto_attribs=True)
class ServoMovement:
    gpin: int = field(default=11)
    position: int = field(default=0)
    servo: GPIO.PWM = field(default=None)

    def __attrs_post_init__(self):
        self.servo = ServoGPIOModule(self.gpin).servo
        self.default_move()

    def _calculate_duty_cycle_percentage(self, angle: int):
        return (angle / 18) + 3

    def default_move(self):
        print(f'---- moving servo to {self.position}----')
        self.servo.ChangeDutyCycle(self.position)
        sleep(0.2)

    def move_to(self, position):
        self.servo.ChangeDutyCycle(position)
        sleep(0.2)

    def stop(self):
        self.servo.ChangeDutyCycle(0)
