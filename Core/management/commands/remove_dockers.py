from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
import os


class Command(BaseCommand):
    def handle(self, *args, **options):
        try:
            os.system('docker container rm mongodb_td --force')
            os.system('docker container rm rabbitmq_td --force')
            self.stdout.write("Docker Container removed!")

            os.system('docker image rm mongo:4.4.8 --force')
            os.system('docker image rm custom-rabbitmq:3.8-management-alpine --force')
            self.stdout.write("Docker Images removed!")
        except Exception as ex:
            print(f'Error {ex} when try to remove information')
