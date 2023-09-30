import time
from django.core.management.base import BaseCommand
import os
import signal


class Command(BaseCommand):
    def ctrl_c_function(self, signum, frame):
        time.sleep(1)
        self.stdout.write("Waiting for Celery...")

    def handle(self, *args, **options):
        signal.signal(signal.SIGINT, self.ctrl_c_function)

        try:
            os.system('gnome-terminal -- sh -c "python -m celery -A Core worker -Q prediction_ok_actions -l info --concurrency=1 --without-gossip --pool=solo; bash"')
            os.system('gnome-terminal -- sh -c "python -m celery -A Core worker -Q purge_data,check_predictions -l info --concurrency=1 --without-gossip --pool=solo; bash"')
        except Exception as ex:
            print(f'Error {ex} when try to start celery')