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

from BackEndApps.Predictions.services.DistanceCalculations import DistanceCalculations, CM_IN_PIXELS


class CameraType(Enum):
    USB = 'usb'
    CSI = 'csi'


class RenderType(Enum):
    CV2 = 'cv2'
    MATPLOTLIB = 'matplotlib'


class SimulateTurretEnv(gym.Env):
    """
    Custom Environment that follows the OpenAI gym interface.
    It simulates a servo motor that needs to center a target within a camera's view.
    """

    def __init__(self):
        super(SimulateTurretEnv, self).__init__()
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

        image = cv2.imread(f'BackEndApps/Predictions/ml_models/images_to_simulate/image_1.jpg')

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
            return self.state, -10, True, {}

        reward = 0
        done = False

        if target_detected:
            self.steps_without_target = 0
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
        else:
            self.steps_without_target += 1

        if self.steps_without_target >= 10:
            print('salimos, no encontramos target')
            self.steps_without_target = 0
            return self.state, self.calculate_reward(), True, {}

        return self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        position = np.clip(randint(1, 10), 1, 12)
        self.state = np.array([position, 0], dtype=np.float32)
        self.steps_without_target = 0
        self.n_stack_at_start = 0
        self.n_stack_at_the_end = 0
        self.regenerate_target = True

        return self.state

    def calculate_reward(self):
        offset = self.state[1]

        reward = -abs(offset)  # Gradual penalty based on distance

        if self.n_stack_at_start >= 4 or self.n_stack_at_the_end >= 4:
            self.n_stack_at_start = 0
            self.n_stack_at_the_end = 0
            reward -= 20  # Adjusted penalty for staying at extremes
        else:
            # Small reward for moving away from extremes
            reward += 2

        if abs(offset) <= 1:
            reward += 15  # Higher reward for centering the target
        elif abs(offset) <= 3:
            reward += 5  # Lower reward for being close

        return reward

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
        return center_x_where_image_was_taken - 1 <= (right + center_target_x) <= center_x_where_image_was_taken + 1

    @staticmethod
    def calculate_offset(calculated_distances: pd.Series):
        left = calculated_distances.left
        right = calculated_distances.right
        from_left_to_center = calculated_distances.from_left_side_to_center
        from_right_to_center = calculated_distances.from_right_side_to_center

        target_width = calculated_distances.width - (right + left)
        center_target_x = target_width / 2
        print(f"left: {left}, right: {right}, center_target_x: {center_target_x}, target_width: {target_width}, target_width: {calculated_distances.width}")
        print(f"from_left_to_center: {from_left_to_center}, from_right_to_center: {from_right_to_center}")
        if right > left:
            return from_right_to_center

        return -from_left_to_center

    def render(self, mode='human'):
        if self.simulated_image is None:
            return

        DistanceCalculations.create_from(
            deepcopy(self.simulated_image), self.simulated_labels
        ).draw_lines_into_image(100, None, False)


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


def discount_rewards(rewards, gamma=0.9):
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


def train_policy_network(
        policy_net: tf.keras.Model,
        optimizer: tf.keras.optimizers,
        episodes=100,
        render_type: RenderType = RenderType.CV2
):
    """
    Trains the policy network.

    Parameters
    ----------
    policy_net : tf.keras.Model
        The policy network model.
    optimizer : tf.keras.optimizers
        The optimizer.
    episodes : int, optional
        The number of episodes to train, by default 100
    render_type : RenderType, optional
        The render type, by default RenderType.CV2
    """
    env_ = SimulateTurretEnv()

    for episode in range(episodes):
        state = env_.reset()
        done = False
        episode_states, episode_actions, episode_rewards = [], [], []

        while not done:
            action = choose_action(policy_net, state)
            print(f"Episode: {episode}, --- action, {action}")

            new_state, reward, done, _ = env_.step(action)
            env_.render()

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
        policy_net.save('./model_binaries/policy_net.h5')

        print(f"Episode: {episode}, Loss: {loss.numpy()}")


if __name__ == '__main__':
    env = SimulateTurretEnv()
    policy_net_def = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(env.observation_space.shape[0],)),
            tf.keras.layers.Dense(8, activation="relu", ),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    train_policy_network(
        tf.keras.models.load_model('../model_binaries/policy_net_main2.h5'),
        tf.optimizers.Adam(learning_rate=0.001)
    )
