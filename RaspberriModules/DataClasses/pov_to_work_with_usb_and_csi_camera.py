import cv2
from CustomPicamera import CustomPicamera
import threading
import time

# Function to capture video from the USB camera using OpenCV
def usb_camera_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Couldn't open the USB camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('USB Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Function to capture video from the Raspberry Pi Camera Module using PiCamera
def csi_camera_feed():
    picamera = CustomPicamera()
    picamera.start_camera()

    while True:
        image = picamera.capture_array()
        cv2.imshow('CSI Camera', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break      

# Run both functions in separate threads
#thread1 = threading.Thread(target=csi_camera_feed)

thread1.start()
cv2.destroyAllWindows()
