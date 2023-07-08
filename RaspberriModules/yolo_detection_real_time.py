import cv2
from picamera2 import MappedArray, Picamera2, Preview
import torch
from DataClasses.ServoModule import ServoMovement

normalSize = (400, 500)

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gerion/TargetDetection/models/yolov5s/exp4/weights/best.pt', force_reload=False) 
cv2.startWindowThread()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": normalSize}, controls={'FrameRate': 30})
picam2.configure(config)
picam2.start_preview(Preview.QTGL)
picam2.start()

frames = 0
angle = 0
gpin_horizontal_servo = 13
increment = 2.5
servo_movements = 0

while True:
    frames += 1
    image = picam2.capture_array()

    if frames < 15:
        continue
    
    print(f'---- add one more movement ---- {image.shape}  {type(image)}')
    servo = ServoMovement(gpin_horizontal_servo, angle)
    servo_movements += 1
    prediction = model(image, augment=True)

    if servo_movements == 4:
        servo.stop()
        increment = -increment
        
    angle += increment

    if angle < 0:
        angle = 0

    frames = 0


