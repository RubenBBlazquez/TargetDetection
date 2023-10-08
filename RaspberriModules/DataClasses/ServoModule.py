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


class ServoGPIOModule:
    def __init__(self, gpin: int) -> None:
        GPIO.setup(gpin, GPIO.OUT)
        self.servo = GPIO.PWM(gpin, 50)
        self.servo_position = -1


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

    def _init_servo(self, gpin: int) -> ServoGPIOModule:
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

        return ServoGPIOModule(gpin)

    def set_actual_servo_position(self, gpin: int, position: int):
        """
        method to set the actual servo position
        if it is different to the last

        Parameters
        ----------
        gpin: int
            physical pin where servo is connected in raspberri pi
        position: int
            position where you want to move the servo
        """
        self.servos[gpin].servo_position = position

    def get_servo(self, gpin: int) -> GPIO.PWM:
        """
        method that initialize or return a servo

        Parameters
        ----------
        gpin: int
            physical pin where servo is connected in raspberri pi
        """
        if self.servos.get(gpin, None):
            return self.servos[gpin]

        new_servo = self._init_servo(gpin)
        self.servos[gpin] = new_servo
        new_servo.servo.start(0)

        return new_servo


@define(auto_attribs=True)
class ServoMovement:
    gpin: int = field(default=11)
    position: int = field(default=0)
    servo_module: ServoGPIOModule = field(default=None)
    name: str = field(default="")
    servo_management: ServoManagement = field(default=ServoManagement())

    def __attrs_post_init__(self):
        self.servo_module = self.servo_management.get_servo(self.gpin)

    def _calculate_duty_cycle_percentage(self, angle: int):
        return (angle / 18) + 3

    def default_move(self):
        if self.servo_module.servo_position == self.position:
            return

        print(f'---- moving servo to {self.position}----')
        self.servo_management.set_actual_servo_position(self.gpin, self.position)
        self.servo_module.servo.ChangeDutyCycle(self.position)
        sleep(0.01)

    def move_to(self, position: int):
        if self.servo_module.servo_position == position:
            return

        print(f'---- moving servo {self.name} to {position}----')
        self.servo_management.set_actual_servo_position(self.gpin, position)
        self.servo_module.servo.ChangeDutyCycle(position)
        sleep(0.01)

    def stop(self):
        self.servo_module.servo.ChangeDutyCycle(0)
