from django.core.management import BaseCommand
from Core.Services.TargetDetection.YoloTargetDetection import YoloTargetDetection


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--directory', type=str, help='path to the directory to train the models')
        parser.add_argument('--epochs', type=int, help='number of epochs to train the yolo model')
        parser.add_argument('--bach_size', type=int, help='number of elements to train in groups each epoch')

    def handle(self, *args, **options):
        directory = options.get('directory', '')
        epochs = options.get('epochs', 50)
        bach_size = options.get('batch_size', 15)

        YoloTargetDetection('best.pt').train(directory, epochs, bach_size)
