import json
import time
from django.core.management.base import BaseCommand
import os
import signal
from Core.settings import directory_separator as slash
PATH_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root


class Command(BaseCommand):
    def ctrl_c_Function(self, signum, frame):
        time.sleep(1)
        os.system('docker start rabbitmq')
        os.system('docker start mongodb5')

    def handle(self, *args, **options):
        signal.signal(signal.SIGINT, self.ctrl_c_Function)

        try:
            os.system(f"docker pull rabbitmq:3.8-management-alpine")
            os.system(f"docker build -t custom-rabbitmq:3.8-management-alpine"
                      f" Core{slash}management{slash}commands{slash}DockerSetUpFiles")
            self.stdout.write("Docker Built!")

            mongo_user = {
                "user": "ruben",
                "pwd": "103856",
                "roles": [
                    {
                        "role": "readWrite",
                        "db": "target_detection"
                    }
                ]
            }

            with open(f'{PATH_DIR}{slash}..{slash}..{slash}..{slash}volumes{slash}mongo-init.js', 'w') as file:
                file.write(f'db.createUser({json.dumps(mongo_user)});')

            os.system('docker compose -p target_detection -f'
                      f'Core{slash}management{slash}commands{slash}DockerSetUpFiles{slash}docker-compose.yml up -d')
            self.stdout.write("Docker Composed!")
        except Exception as ex:
            print(f'Error {ex} when try to create dockers')
