import platform
from django.core.management.base import BaseCommand
from BackEndApps.Predictions.ml_models.real_turret_env import start_model_detection
from BackEndApps.Predictions.ml_models.simulate_turret_env.simulate_turret_env import train_policy_network, \
    SimulateTurretEnv, RenderType
from BackEndApps.Predictions.CeleryTasks import real_time_detection_with_celery
from enum import Enum
import tensorflow as tf


class CameraType(Enum):
    USB = 'usb'
    CSI = 'csi'


class Command(BaseCommand):
    @staticmethod
    def detection_with_celery():
        real_time_detection_with_celery()

    @staticmethod
    def detection_with_ml_model(simulate_training=False, trained_model=''):
        if simulate_training:
            policy_net_def = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        10,
                        activation="relu",
                        input_shape=(SimulateTurretEnv().observation_space.shape[0],)
                    ),
                    tf.keras.layers.Dense(8, activation="relu", ),
                    tf.keras.layers.Dense(3, activation="softmax"),
                ]
            )
            train_policy_network(
                policy_net_def,
                tf.optimizers.Adam(learning_rate=0.01),
                episodes=1000,
                render_type=(
                    RenderType.MATPLOTLIB
                    if platform.system().lower() == 'linux'
                    else RenderType.CV2
                )
            )
            return

        if not trained_model:
            raise ValueError('No trained model was provided')

        start_model_detection(
            tf.keras.models.load_model(f"BackEndApps/Predictions/ml_models/model_binaries/{trained_model}")
        )

    def add_arguments(self, parser):
        parser.add_argument('--use_celery', type=bool, default=False, help='path to the directory to train the models')
        parser.add_argument('--simulate_training', type=bool, default=False, help='boolean to simulate training or not')
        parser.add_argument('--trained_model', type=str, default='', help='name of the trained model')

    def handle(self, *args, **options):
        use_celery = options.get('use_celery')

        if use_celery:
            self.detection_with_celery()
            return

        simulate_training = options.get('simulate_training')
        trained_model = options.get('trained_model')
        self.detection_with_ml_model(simulate_training, trained_model)
