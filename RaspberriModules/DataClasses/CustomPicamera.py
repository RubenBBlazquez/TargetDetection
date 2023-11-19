from picamera2 import Picamera2, Preview, picamera2
from attrs import define, field


class CustomPicamera(Picamera2):
    camera_size: tuple = (400, 500)
    frame_rate = 30
    preview_type = Preview.QTGL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.create_preview_configuration(main={"size": self.camera_size})
        self.configure(config)

    def start_camera(self):
        self.start()
