import os

from django.core.management import BaseCommand

from BackEndApps.DataTraining.Services.Trainers.MessagesCollectors.yolo_train_message_collector import \
    YoloTrainMessageCollector
from BackEndApps.DataTraining.Services.Trainers.yolo_trainer import YoloTrainer
from BackEndApps.DataTraining.Services.Uploaders.local_model_uploader import LocalModelUploader
from Core.settings import directory_separator as slash


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--yolo_file_data', type=str, help='path to the directory to train the models')
        parser.add_argument('--epochs', type=int, help='number of epochs to train the yolo model')
        parser.add_argument('--bach_size', type=int, help='number of elements to train in groups each epoch')
        parser.add_argument('--pretrained_model_path', type=str, help='path where the model weights are saved')

    def handle(self, *args, **options):
        yolo_file_data = options.get('yolo_file_data', '')
        epochs = options.get('epochs', 50)
        bach_size = options.get('batch_size', 15)
        pretrained_model_path = options.get('pretrained_model_path', 'models/best.pt')

        trainer = YoloTrainer(
            LocalModelUploader(new_path=f'models{slash}'),
            YoloTrainMessageCollector(),
            **{
                'yolo_file_data': yolo_file_data,
                'epochs': epochs,
                'batch_size': bach_size,
                'pretrained_model_path': pretrained_model_path
            }
        )
        trainer.train()