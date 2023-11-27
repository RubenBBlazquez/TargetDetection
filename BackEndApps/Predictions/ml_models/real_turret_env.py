from copy import deepcopy
import os
import time
from enum import Enum
from random import randint
from typing import Any

import cv2
import gym
import pandas as pd
from gym import spaces
import numpy as np
import tensorflow as tf

from BackEndApps.Predictions.services.DistanceCalculations import CM_IN_PIXELS, DistanceCalculations
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection


class CameraType(Enum):
    USB = 'usb'
    CSI = 'csi'


def get_image_from_camera(camera: Any, camera_type: CameraType, usb_camera_port=0):
    if camera_type == CameraType.CSI:
        return camera.capture_array()

    ret, image = camera.read()

    if type(image) != np.ndarray:
        usb_camera_port = 1 if usb_camera_port == 0 else 0
        camera = cv2.VideoCapture(
            usb_camera_port
        )
        ret, image = camera.read()

    ret.release()
    return image


class RealTurretEnv(gym.Env):
    """
    Custom Environment that follows the OpenAI gym interface.
    It simulates a servo motor that needs to center a target within a camera's view.
    """

    def __init__(self):
        super(RealTurretEnv, self).__init__()

        # we import here to avoid errors when we are not execution on the raspberry pi
        from RaspberriModules.DataClasses.CustomPicamera import CustomPicamera

        cv2.startWindowThread()

        self.servo = None
        self.image = None
        self.labels = pd.Series()
        self.picamera = CustomPicamera()
        self.picamera.start()
        self.usb_camera_port = 0
        self.model = YoloTargetDetection(os.getenv('YOLO_MODEL_NAME'))
        self.model.predict(cv2.imread('BackEndApps/Predictions/ml_models/images_to_simulate/aaa.jpg'))

        self.usb_camera = cv2.VideoCapture(
            self.usb_camera_port
        )
        self.n_stack_at_the_end = 0
        self.n_stack_at_start = 0

        self.n_stay_actions = 0

        # Define action and observation space
        # Actions: Move left (-1), stay (0), move right (+1)
        self.action_space = spaces.Discrete(3)

        # Observations: Position of the motor, and the offset of the target from the center
        # Here, we consider the offset as a signed value to indicate the direction
        self.observation_space = spaces.Box(low=np.array([2, -12]), high=np.array([12, 12]), dtype=np.float32)

        # Initial state: [motor position, target offset]
        self.state = np.array([0, 0], dtype=np.float32)

        self.steps_without_target = 0

    def update_target(self, movement: int = 0) -> None:
        x_center_update = (2 * CM_IN_PIXELS) * movement
        self.labels.xcenter = np.clip(
            self.labels.xcenter + x_center_update, 0, self.image.shape[1]
        )

    def step(self, action) -> Any:
        # Decode action
        if action == 0:
            movement = -1  # Move left
            self.n_stay_actions = 0
        elif action == 1:
            movement = 1  # Move right
            self.n_stay_actions = 0
        else:
            self.n_stay_actions += 1
            movement = 0  # Stay

        # Update state: move the motor
        self.state[0] = np.clip(self.state[0] + movement, 2, 12)
        self.servo.move_to(self.state[0])
        time.sleep(0.3)
        self.servo.stop()
        self.update_target()

        if self.state[0] >= 11:
            self.n_stack_at_the_end += 1
        else:
            self.n_stack_at_the_end = 0

        if self.state[0] <= 2:
            self.n_stack_at_start += 1
        else:
            self.n_stack_at_start = 0

        self.steps_without_target = 0
        is_target_out_bounds = (
                (self.n_stack_at_the_end >= 4 and self.state[0] == 12 and movement == 1) or
                (self.n_stack_at_start >= 4 and self.state[0] == 1 and movement == -1)
        )
        if is_target_out_bounds:
            print('salimos, debido a que el objetivo está fuera de los límites')
            return self.state, True

        distance_calculations = DistanceCalculations.create_from(self.image, self.labels)
        calculated_distances = distance_calculations.get_all_distances()

        # Simulate target detection and calculate the new offset
        # Here, you would integrate with your computer vision system
        # For this example, we simulate the target offset update
        self.state[1] = self.calculate_offset(calculated_distances)

        # Check if the episode is done
        return self.state, self.n_stay_actions >= 3

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.array([2, 0], dtype=np.float32)
        time.sleep(1)
        self.steps_without_target = 0
        self.n_stack_at_start = 0
        self.n_stack_at_the_end = 0
        self.n_stay_actions = 0

        return self.state

    @staticmethod
    def calculate_offset(calculated_distances: pd.Series):
        left = calculated_distances.left
        right = calculated_distances.right

        center = (calculated_distances.width / 2) - 1
        target_width = calculated_distances.width - (right + left)
        center_target_x = target_width / 2

        if right > left:
            return -(center + left + center_target_x)

        return center + right + center_target_x

    def render(self, mode='human'):
        print(self.labels)
        DistanceCalculations.create_from(
            deepcopy(self.image), self.labels
        ).draw_lines_into_image(100)

        cv2.waitKey(500)


def choose_action(model, state):
    """
    Chooses an action according to the policy network's output probabilities.
    Args:
    - model (tf.keras.Model): The policy network model.
    - state (numpy.array): The current state.

    Returns:
    - action (int): Selected action.
    """
    state_input = tf.convert_to_tensor([state], dtype=tf.float32)
    action_probs = model(state_input)
    action = tf.random.categorical(tf.math.log(action_probs), 1).numpy()[0, 0]
    return action


def start_model_detection(policy_net: tf.keras.Model) -> None:
    """
    Starts the model detection.

    Parameters
    ----------
    policy_net : tf.keras.Model
        The policy network model.

    Returns
    -------
    None.
    """
    env_ = RealTurretEnv()
    cv2.startWindowThread()

    while True:
        state = env_.reset()
        done = False
        position = 2
        movement = 1

        while True:
            image = get_image_from_camera(env_.usb_camera, CameraType.USB, env_.usb_camera_port)
            cv2.imshow("Camera", image)
            cv2.waitKey(1)

            result = env_.model.predict(image)
            print(result.boxes)

            if len(result) and result.boxes.conf[0] > 0.6:
                xywh = result.boxes.xywh[0]
                labels = pd.Series(
                    {
                        'xcenter': xywh[0],
                        'ycenter': xywh[1],
                        'width': xywh[2],
                        'height': xywh[3],
                        'confidence': result.boxes.conf[0],
                    }
                )
                env_.image = image
                env_.labels = labels
                env_.state = [position, 0]
                break

            env_.servo.move_to(position)
            time.sleep(0.3)
            env_.servo.stop()

            movement_multiplier = movement * randint(1, 3)
            position = np.clip(
                position + movement_multiplier,
                2, 12
            )
            print(position, movement)

            if position >= 12 or position <= 2:
                movement *= -1

        while not done:
            action = choose_action(policy_net, state)
            print('action', action)
            new_state = env_.step(action)
            env_.render()
            print("-----------------------------------------------------")
            print(new_state)
            print("-----------------------------------------------------")

            state = new_state


if __name__ == '__main__':
    start_model_detection(tf.keras.models.load_model("./model_binaries/policy_net_main2.h5"))