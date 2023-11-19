import os
import time
from enum import Enum
from random import randint
import cv2
import gym
import pandas as pd
from gym import spaces
import numpy as np
import tensorflow as tf

from BackEndApps.Predictions.services.DistanceCalculations import DistanceCalculations
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection
from RaspberriModules.DataClasses.CustomPicamera import CustomPicamera
from RaspberriModules.DataClasses.ServoModule import ServoMovement


class CameraType(Enum):
    USB = 'usb'
    CSI = 'csi'


class ServoMotorEnv(gym.Env):
    """
    Custom Environment that follows the OpenAI gym interface.
    It simulates a servo motor that needs to center a target within a camera's view.
    """

    def __init__(self):
        super(ServoMotorEnv, self).__init__()
        cv2.startWindowThread()

        self.servo = ServoMovement(int(os.getenv('X_SERVO_PIN')), 1, name='x1')
        # self.picamera = CustomPicamera()
        # self.picamera.start()
        self.usb_camera_port = 0
        self.model = YoloTargetDetection(os.getenv('YOLO_MODEL_NAME'))
        self.usb_camera = cv2.VideoCapture(
            self.usb_camera_port
        )
        self.n_stack_at_the_end = 0
        self.n_stack_at_start = 0

        # Define action and observation space
        # Actions: Move left (-1), stay (0), move right (+1)
        self.action_space = spaces.Discrete(3)

        # Observations: Position of the motor, and the offset of the target from the center
        # Here, we consider the offset as a signed value to indicate the direction
        self.observation_space = spaces.Box(low=np.array([0, -12]), high=np.array([12, 12]), dtype=np.float32)

        # Initial state: [motor position, target offset]
        self.state = np.array([0, 0], dtype=np.float32)

        self.steps_without_target = 0

    def get_image_from_camera(self, camera_type: CameraType, init_cam=True):
        if camera_type == CameraType.CSI:
            return self.picamera.capture_array()

        ret, image = self.usb_camera.read()

        if type(image) != np.ndarray:
            self.usb_camera_port = 1 if self.usb_camera_port == 0 else 0
            self.usb_camera = cv2.VideoCapture(
                self.usb_camera_port
            )
            ret, image = self.usb_camera.read()

        return image

    def step(self, action) -> tuple:
        image = self.get_image_from_camera(CameraType.USB, True)
        result = self.model.predict(image)
        labels = result.pandas().xywh[0]
        predicted_labels = labels[labels['confidence'] > 0.60]
        target_detected = not predicted_labels.empty
        print(labels, target_detected)

        if not target_detected:
            self.steps_without_target += 1
        else:
            self.steps_without_target = 0

        # Decode action
        if action == 0:
            movement = -1  # Move left
        elif action == 1:
            movement = 1  # Move right
        else:
            movement = 0  # Stay

        # Update state: move the motor
        self.state[0] = np.clip(self.state[0] + movement, 1, 12)
        self.servo.move_to(int(self.state[0]))
        time.sleep(0.1)

        if self.state[0] >= 11:
            self.n_stack_at_the_end += 1
        else:
            self.n_stack_at_the_end = 0

        if self.state[0] <= 2:
            self.n_stack_at_start += 1
        else:
            self.n_stack_at_start = 0

        reward = 0
        done = False
        
        if target_detected:
            distance_calculations = DistanceCalculations.create_from(image, labels)
            calculated_distances = distance_calculations.get_all_distances()

            # Simulate target detection and calculate the new offset
            # Here, you would integrate with your computer vision system
            # For this example, we simulate the target offset update
            self.state[1] = self.calculate_offset(calculated_distances)

            # Calculate reward
            reward = self.calculate_reward()

            # Check if the episode is done
            done = self.is_target_centered(calculated_distances)

        if self.steps_without_target > 12:
            print('salimos, no encontramos target')
            self.steps_without_target = 0
            return self.state, self.calculate_reward(), True, {}

        return self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        position = np.clip(randint(1, 6), 1, 12)
        self.state = np.array([position, 0], dtype=np.float32)
        self.servo.move_to(position)
        time.sleep(1)
        self.steps_without_target = 0
        self.n_stack_at_start = 0
        self.n_stack_at_the_end = 0

        return self.state

    def calculate_reward(self):
        offset = self.state[1]

        if self.n_stack_at_start >= 4 or self.n_stack_at_the_end >= 4:
            return -50  # Adjusted penalty

        reward = -abs(offset)  # Gradual penalty based on distance

        if abs(offset) <= 1:  
            reward += 15  # Adjusted reward for centering the target

        return reward


    def is_target_centered(self, calculated_distances: pd.Series):
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
        return center_x_where_image_was_taken - 2 <= (right + center_target_x) <= center_x_where_image_was_taken + 2

    def calculate_offset(self,calculated_distances: pd.Series):
        left = calculated_distances.left
        right = calculated_distances.right

        # Define the condition for ending the episode
        target_width = calculated_distances.width - (right + left)
        center_target_x = target_width / 2

        if right > left:
            return left + center_target_x

        return right + center_target_x


    def render(self, mode='human'):
        # Render the environment to the screen (optional)
        print(f"Servo Position: {self.state[0]}, Target Offset: {self.state[1]}")


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


def discount_rewards(rewards, gamma=0.95):
    """
    Take 1D float array of rewards and compute discounted rewards.
    Args:
    - rewards (numpy.array): Rewards at each time step.
    - gamma (float): Discount factor.

    Returns:
    - discounted_rewards (numpy.array): The discounted rewards.
    """
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def train_policy_network(env: ServoMotorEnv, policy_net, optimizer, episodes=100):
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_states, episode_actions, episode_rewards = [], [], []

        while not done:
            action = choose_action(policy_net, state)
            print('action', action)
            new_state, reward, done, _ = env.step(action)

            print("-----------------------------------------------------")
            print(new_state, reward, done)
            print("-----------------------------------------------------")


            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = new_state

        discounted_rewards = discount_rewards(episode_rewards)

        states_tensor = tf.convert_to_tensor(episode_states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(episode_actions, dtype=tf.int32)
        rewards_tensor = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

        with tf.GradientTape() as tape:
            action_probs = policy_net(states_tensor)
            action_masks = tf.one_hot(actions_tensor, action_probs.shape[1])
            masked_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(masked_probs) * rewards_tensor)

        gradients = tape.gradient(loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables))

        print(f"Episode: {episode}, Loss: {loss.numpy()}")

