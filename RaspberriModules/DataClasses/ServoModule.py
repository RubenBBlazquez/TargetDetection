from time import sleep
from attrs import define, field
import RPi.GPIO as GPIO
from typing import Dict

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class ServoGPIOModule():
    def __init__(self, gpin: int) -> None:
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(gpin, GPIO.OUT)
        self.servo = GPIO.PWM(gpin, 50)

@define(auto_attribs=True)
class ServoManagement(metaclass=SingletonMeta):
    """
    Class to manage the initialization of the servos

    Parameters
    ----------
    servos: Dict[int, GPIO.PWM]
        contains all the servos initialized
    
    Notes
    -----
    This class is necesary because if you want to initialize a servo twice, 
    you get a runtime error

    """

    servos: Dict[int, GPIO.PWM] = field(default={})

    def _init_servo(self, gpin: int) -> GPIO:
        """
        method to manage the initialization of the servos

        Parameters
        ----------
        gpin: int
            physical pin where servo is connected in raspberri pi
        
        Returns
        --------
            the servo initialized to start working
    
        """
        if len(self.servos.keys()) == 0:
            GPIO.cleanup()
            GPIO.setmode(GPIO.BOARD)
        
        GPIO.setup(gpin, GPIO.OUT)
        return GPIO.PWM(gpin, 50)

    def get_servo(self, gpin: int) -> GPIO.PWM:
        if self.servos.get(gpin, None):
            return self.servos[gpin]

        new_servo = self._init_servo(gpin)
        self.servos[gpin] = new_servo
        new_servo.start(0)

        return new_servo

@define(auto_attribs=True)
class ServoMovement:
    gpin: int = field(default=11)
    position: int = field(default=0)
    servo: GPIO.PWM = field(default=None)

    def __attrs_post_init__(self):
        servo = ServoManagement().get_servo(self.gpin)
        self.servo = servo

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
