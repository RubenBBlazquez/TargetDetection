import os

from django.core.management import BaseCommand
from Core.settings import directory_separator as slash

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--dataset_1', type=str, help='path to the directory of the first dataset')
        parser.add_argument('--dataset_2', type=str, help='path to the directory of the second dataset')
        parser.add_argument('--model_export', type=str, help='path to the directory to export the model')
        parser.add_argument('--prefix', type=int, help='prefix to add to the labels of the second dataset')

    def handle(self, *args, **options):
        dataset_1 = options.get('dataset_1', '')
        dataset_2 = options.get('dataset_2', '')
        model_export = options.get('model_export', '')
        prefix = options.get('prefix', 0)
        model_export = model_export if model_export.endswith('/') else f'{model_export}/'
        dataset_1 = '/'.join(dataset_1.split('/')[-1]) if dataset_1.endswith('/') else dataset_1
        dataset_2 = '/'.join(dataset_2.split('/')[-1]) if dataset_2.endswith('/') else dataset_2

        train_directory = f'{slash}train{slash}'
        valid_directory = f'{slash}valid{slash}'
        test_directory = f'{slash}test{slash}'

        directories = [train_directory, valid_directory, test_directory]

        for directory in directories:
            image_directory = f'{slash}/images{slash}'
            label_directory = f'{slash}/labels{slash}'

            images_1 = [
                image for image in os.listdir(f'{dataset_1}{directory}{image_directory}')
            ]
            labels = [
                labels for labels in os.listdir(f'{dataset_1}{directory}{label_directory}')
            ]
            images_2 = [
                image for image in os.listdir(f'{dataset_2}{directory}{image_directory}')
            ]
            labels_2 = [
                labels for labels in os.listdir(f'{dataset_2}{directory}{label_directory}')
            ]


            for image in images_2:
                print(image)
                # os.rename(image, f'{dataset_2}{train_directory}{image_directory}{prefix}_{image}')

            for label in labels_2:
                print(label)
                # os.rename(label, f'{dataset_2}{train_directory}{image_directory}{prefix}_{label}')
