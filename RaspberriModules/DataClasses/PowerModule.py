from attrs import define, field
import RPi.GPIO as GPIO


@define(auto_attribs=True)
class PowerModule:
    gpin: int = field(default=15)

    def on(self):
        GPIO.setup(self.gpin, GPIO.OUT)
        GPIO.output(self.gpin, GPIO.HIGH)

    def off(self):
        GPIO.setup(self.gpin, GPIO.OUT)
        GPIO.output(self.gpin, GPIO.LOW)
