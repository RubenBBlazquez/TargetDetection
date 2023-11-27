import time
from enum import Enum
from random import randint
from typing import Tuple
from copy import deepcopy
import cv2
import gym
import pandas as pd
from gym import spaces
import numpy as np
import tensorflow as tf

from BackEndApps.Predictions.ml_models.real_turret_env import get_image_from_camera
from BackEndApps.Predictions.services.DistanceCalculations import DistanceCalculations, CM_IN_PIXELS


class CameraType(Enum):
    USB = 'usb'
    CSI = 'csi'


class RenderType(Enum):
    CV2 = 'cv2'
    MATPLOTLIB = 'matplotlib'


class CheckSimulationTurretEnv(gym.Env):
    """
    Custom Environment that follows the OpenAI gym interface.
    It simulates a servo motor that needs to center a target within a camera's view.
    """

    def __init__(self):
        super(CheckSimulationTurretEnv, self).__init__()
        cv2.startWindowThread()

        self.n_stack_at_the_end = 0
        self.n_stack_at_start = 0
        self.steps_without_target = 0
        self.regenerate_target = True
        self.simulated_image = None
        self.simulated_labels = None
        self.render_method = RenderType.CV2

        # Define action and observation space
        # Actions: Move left (-1), stay (0), move right (+1)
        self.action_space = spaces.Discrete(3)

        # Observations: Position of the motor, and the offset of the target from the center
        # Here, we consider the offset as a signed value to indicate the direction
        self.observation_space = spaces.Box(low=np.array([0, -12]), high=np.array([12, 12]), dtype=np.float32)

        # Initial state: [motor position, target offset]
        self.state = np.array([0, 0], dtype=np.float32)

    def simulate_information(self, movement: int = 0) -> Tuple[bool, np.ndarray, pd.Series]:
        if not self.regenerate_target:
            x_center_update = (2 * CM_IN_PIXELS) * movement
            self.simulated_labels.xcenter = np.clip(
                self.simulated_labels.xcenter + x_center_update, 0, self.simulated_image.shape[1]
            )

            return True, self.simulated_image, self.simulated_labels

        image = cv2.imread(f'../images_to_simulate/image_1.jpg')

        if type(image) != np.ndarray:
            return False, np.ndarray([]), pd.Series()

        self.regenerate_target = False
        random_division = randint(5, 10)
        random_width = image.shape[1] // random_division
        random_height = image.shape[0] // random_division
        random_x_center = randint(0, image.shape[1] - random_width)
        random_y_center = randint(0, image.shape[0] - random_height)

        self.simulated_labels = pd.Series({
            'xcenter': random_x_center,
            'ycenter': random_y_center,
            'width': random_width,
            'height': random_height
        })
        self.simulated_image = deepcopy(image)

        return True, image, self.simulated_labels

    def step(self, action) -> tuple:
        # Decode action
        if action == 0:
            movement = -1  # Move left
        elif action == 1:
            movement = 1  # Move right
        else:
            movement = 0  # Stay

        target_detected, image, labels = self.simulate_information(movement)

        # Update state: move the motor
        self.state[0] = np.clip(self.state[0] + movement, 1, 12)

        if self.state[0] >= 11:
            self.n_stack_at_the_end += 1
        else:
            self.n_stack_at_the_end = 0

        if self.state[0] <= 2:
            self.n_stack_at_start += 1
        else:
            self.n_stack_at_start = 0

        if (
                (self.n_stack_at_the_end >= 4 and self.state[0] == 12 and movement == 1) or
                (self.n_stack_at_start >= 4 and self.state[0] == 1 and movement == -1)
        ):
            print('salimos, debido a que el objetivo está fuera de los límites')
            return self.state, True

        done = False

        if target_detected:
            self.steps_without_target = 0
            distance_calculations = DistanceCalculations.create_from(image, labels)
            calculated_distances = distance_calculations.get_all_distances()

            # Simulate target detection and calculate the new offset
            # Here, you would integrate with your computer vision system
            # For this example, we simulate the target offset update
            self.state[1] = self.calculate_offset(calculated_distances)

            # # Check if the episode is done
            # done = self.is_target_centered(calculated_distances)
        else:
            self.steps_without_target += 1

        if self.steps_without_target >= 10:
            print('salimos, no encontramos target')
            self.steps_without_target = 0
            return self.state, True

        return self.state, done,

    def reset(self):
        position = np.clip(randint(1, 10), 1, 12)
        self.state = np.array([position, 0], dtype=np.float32)
        self.steps_without_target = 0
        self.n_stack_at_start = 0
        self.n_stack_at_the_end = 0
        self.regenerate_target = True

        return self.state

    @staticmethod
    def calculate_offset(calculated_distances: pd.Series):
        left = calculated_distances.left
        right = calculated_distances.right

        target_width = calculated_distances.width - (right + left)
        center_target_x = target_width / 2
        breakpoint()
        if right > left:
            return -(left + center_target_x)

        return right + center_target_x

    @staticmethod
    def is_target_centered(calculated_distances: pd.Series):
        left = calculated_distances.left
        right = calculated_distances.right

        # Define the condition for ending the episode
        target_width = calculated_distances.width - (right + left)
        center_target_x = target_width / 2
        center_x_where_image_was_taken = (calculated_distances.width / 2) - 1

        # we are using this formula to calculate if the target is centered:
        # (center_image - 0.5 <= (right + center_target_x) <= center_image + 0.5)
        # if we have the image (witdh = 100) and (left = 44.50) and (right = 45) and the target has 10 cm size
        # so the formula to know if is centered is: 50 - 1 <= (45 + 5) <= 50 + 1 = true
        return center_x_where_image_was_taken - 0.5 <= (right + center_target_x) <= center_x_where_image_was_taken + 0.5

    def render(self, mode='human'):
        if self.simulated_image is None:
            return

        DistanceCalculations.create_from(
            deepcopy(self.simulated_image), self.simulated_labels
        ).draw_lines_into_image(10000, None, True)


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
    env_ = CheckSimulationTurretEnv()
    cv2.startWindowThread()

    while True:
        state = env_.reset()
        done = False

        while not done:
            action = choose_action(policy_net, state)
            print('action', action)
            new_state, done = env_.step(action)
            env_.render()
            print("-----------------------------------------------------")
            print(new_state)
            print("-----------------------------------------------------")

            state = new_state


if __name__ == '__main__':
    model = tf.keras.models.load_model(f"..\\model_binaries\\policy_net_main2.h5")
    start_model_detection(model)
