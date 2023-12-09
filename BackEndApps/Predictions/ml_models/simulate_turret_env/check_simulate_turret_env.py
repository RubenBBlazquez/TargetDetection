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
        self.last_offset = 0

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
            x_center_update = CM_IN_PIXELS * movement
            print(f"self.simulated_labels.xcenter: {self.simulated_labels.xcenter}")
            self.simulated_labels.xcenter = np.clip(
                self.simulated_labels.xcenter - x_center_update, 0, self.simulated_image.shape[1]
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
        if action == 0:
            movement = -1  # Move left
        elif action == 1:
            movement = 1  # Move right
        else:
            movement = 0  # Stay

        target_detected, image, labels = self.simulate_information(movement)

        if target_detected:
            self.steps_without_target = 0
            distance_calculations = DistanceCalculations.create_from(image, labels)
            calculated_distances = distance_calculations.get_all_distances()

            self.state[1] = self.calculate_offset(calculated_distances)
            self.state[0] = np.clip(self.state[0] + 0.5 * movement, 0.5, 12)

            if self.state[0] >= 12:
                self.n_stack_at_the_end += 1
            else:
                self.n_stack_at_the_end = 0

            if self.state[0] <= 1:
                self.n_stack_at_start += 1
            else:
                self.n_stack_at_start = 0

            if (
                    self.n_stack_at_the_end >= 4 and self.state[0] == 12 and movement == 1 and labels.xcenter >
                    image.shape[0] - 50
            ) or (
                    self.n_stack_at_start >= 4 and self.state[0] == 1 and movement == -1 and labels.xcenter < 50
            ):
                print('salimos, debido a que el objetivo está fuera de los límites')
                return self.state, True

            # Calculate reward
            reward = self.calculate_reward(action)

            print("1 -----------------------------------------------------")
            print(self.state, reward, self.is_motor_going_to_the_right_direction(action))
            print("-----------------------------------------------------")

            self.last_offset = self.state[1]

            # Check if the episode is done
            done = self.is_target_centered(calculated_distances)

            return self.state, done

        self.steps_without_target += 1
        if self.steps_without_target >= 10:
            print('salimos, no encontramos target')
            self.steps_without_target = 0
            return self.state, True

    def reset(self):
        # Reset the state of the environment to an initial state
        random_position = 0.5 * randint(1, 24)
        position = np.clip(random_position, 0.5, 12)
        self.state = np.array([position, 0], dtype=np.float32)
        self.steps_without_target = 0
        self.n_stack_at_start = 0
        self.n_stack_at_the_end = 0
        self.regenerate_target = True
        self.last_offset = None

        return self.state

    def is_motor_going_to_the_right_direction(self, action: int) -> bool:
        """
        This method is used to know if the motor is going to the right direction

        Parameters
        ----------
        action : int
            This is the action that the motor is going to do.
        """
        if not self.last_offset:
            return True

        return abs(self.state[1]) < abs(self.last_offset)

    def calculate_reward(self, action: int):
        """
        Calculates the reward for the current step.

        Parameters
        ----------
        action : int
            This is the action that the motor is going to do.
        """
        offset = self.state[1]

        reward = -abs(offset)  # Gradual penalty based on distance

        if self.n_stack_at_the_end >= 4 or self.n_stack_at_start >= 4:
            reward -= 20

        # Extra penalty for moving in the wrong direction
        if not self.is_motor_going_to_the_right_direction(action):
            offset_difference = abs(self.last_offset) - abs(offset)
            print(f"offset_difference: {offset_difference} offset: {self.state[1]} last_offset: {self.last_offset}")
            reward -= abs(offset_difference * 3)

        if abs(offset) <= 1:
            reward += 10  # Higher reward for centering the target
        elif abs(offset) <= 3:
            reward += 3  # Lower reward for being close

        return reward

    @staticmethod
    def is_target_centered(calculated_distances: pd.Series):
        """
        This method is used to know if the target is centered

        Parameters
        ----------
        calculated_distances : pd.Series
            This is the calculated distances of the target.
        """
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
        return center_x_where_image_was_taken - 1 <= (right + center_target_x) <= center_x_where_image_was_taken + 1

    @staticmethod
    def calculate_offset(calculated_distances: pd.Series):
        """
        This method is used to calculate the offset of the target

        Parameters
        ----------
        calculated_distances : pd.Series
            This is the calculated distances of the target.
        """
        left = calculated_distances.left
        right = calculated_distances.right
        from_x_center_to_center_image = calculated_distances.from_x_center_to_center_image

        if right > left:
            return from_x_center_to_center_image

        return -from_x_center_to_center_image

    def render(self, mode='human'):
        """
        This method is used to render the image

        Parameters
        ----------
        mode : str, optional
            This is the mode that we want to use to render the image, by default 'human'
        """
        if self.simulated_image is None:
            return

        DistanceCalculations.create_from(
            deepcopy(self.simulated_image), self.simulated_labels
        ).draw_lines_into_image(
            100,
            None,
            self.render_method == RenderType.CV2
        )


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
    env_.reset()
    done = False
    state = env_.state

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
    model = tf.keras.models.load_model(f"..\\model_binaries\\policy_net.h5")
    start_model_detection(model)
