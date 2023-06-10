import os
from celery import Celery

# if you want to launch celery worker, you must use the name Core
# for example celery -A Core worker
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Core.settings')

app = Celery('TargetDetection')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()