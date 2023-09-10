from django.core.management import BaseCommand

from BackEndApps.DataTraining.Services.Trainers.yolo_trainer import YoloTrainer
from BackEndApps.DataTraining.Services.Uploaders.base_model_uploader import DummyModelUploader
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--yolo_file_data', type=str, help='path to the directory to train the models')
        parser.add_argument('--epochs', type=int, help='number of epochs to train the yolo model')
        parser.add_argument('--bach_size', type=int, help='number of elements to train in groups each epoch')

    def handle(self, *args, **options):
        yolo_file_data = options.get('yolo_file_data', '')
        epochs = options.get('epochs', 50)
        bach_size = options.get('batch_size', 15)

        YoloTrainer(DummyModelUploader()).train(**{
            'yolo_file_data': yolo_file_data,
            'epochs': epochs,
            'batch_size': bach_size
        })