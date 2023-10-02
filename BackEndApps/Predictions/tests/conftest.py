from unittest.mock import Mock, MagicMock
import sys

# Create a mock package for RPi
RPi_mock = MagicMock()
sys.modules['RPi'] = RPi_mock

# Create a mock module for RPi.GPIO
gpio_mock = Mock()
RPi_mock.GPIO = gpio_mock
sys.modules['RPi.GPIO'] = gpio_mock

def pytest_sessionstart(session):
    """
    This function is executed before the test session starts
    """
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BackEndApps.settings')