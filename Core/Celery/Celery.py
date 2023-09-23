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
    Queue('check_predictions', Exchange('check_predictions'), routing_key='check_predictions',
          queue_arguments={'x-max-priority': 10}),
    Queue('purge_data', Exchange('purge_data'), routing_key='purge_data',
          queue_arguments={'x-max-priority': 10}),
    Queue('prediction_ok_actions', Exchange('prediction_ok_actions'), routing_key='prediction_ok_actions',
          queue_arguments={'x-max-priority': 10}),
]
