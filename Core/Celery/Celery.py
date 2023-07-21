import os
from celery import Celery
from kombu import Queue, Exchange

# if you want to launch celery worker, you must use the name Core
# for example celery -A Core worker
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Core.settings')

app = Celery('TargetDetection')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks(['BackEndApps.Predictions'], related_name='CeleryTasks')

app.conf.task_queues = [
    Queue('YoloPredictions', Exchange('default'), routing_key='default', queue_arguments={'x-max-priority': 10})
]
