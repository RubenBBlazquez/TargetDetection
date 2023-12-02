import os
import shutil

import cv2
from django.core.management import BaseCommand
from Core.settings import directory_separator as slash


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--dataset_1', type=str, help='path to the directory of the first dataset', default='')
        parser.add_argument('--dataset_2', type=str, help='path to the directory of the second dataset', default='')
        parser.add_argument(
            '--copy_in',
            type=int,
            help='set the dataset for copying the images and labels , 1 or 2 ....',
            default=1
        )

    def handle(self, *args, **options):
        dataset_1 = options.get('dataset_1')
        dataset_2 = options.get('dataset_2')
        copy_in = options.get('copy_in')
        print(dataset_1, dataset_2, copy_in)
        dataset_1 = '/'.join(dataset_1.split('/')[-1]) if dataset_1.endswith('/') else dataset_1
        dataset_2 = '/'.join(dataset_2.split('/')[-1]) if dataset_2.endswith('/') else dataset_2

        train_directory = f'{slash}train{slash}'
        valid_directory = f'{slash}valid{slash}'
        test_directory = f'{slash}test{slash}'

        directories = [train_directory, valid_directory, test_directory]

        for directory in directories:
            d1_directory_path = f'{dataset_1}{directory}'
            d2_directory_path = f'{dataset_2}{directory}'
            image_directory_1 = f'{d1_directory_path}images{slash}'
            label_directory_1 = f'{d1_directory_path}labels{slash}'
            image_directory_2 = f'{d2_directory_path}images{slash}'
            label_directory_2 = f'{d2_directory_path}labels{slash}'

            images_1 = os.listdir(f'{image_directory_1}')
            images_2 = os.listdir(f'{image_directory_2}')

            new_width, new_height = cv2.imread(f'{image_directory_1}{images_1[0]}').shape[:2]

            # information where we are going to copy the images and labels
            images = images_2 if copy_in == 1 else images_1
            image_directory = image_directory_2 if copy_in == 1 else image_directory_1
            label_directory = label_directory_2 if copy_in == 1 else label_directory_1

            for image_file in images:
                image = cv2.imread(f'{image_directory}{image_file}')
                original_width, original_height = image.shape[:2]
                image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # we suppose that the labels have the same name as the images (in yolo is the same)
                labels_file = image_file.replace('.jpg', '.txt')
                labels_file = labels_file.replace('.png', '.txt')
                labels_file_io = open(f'{label_directory}{labels_file}', 'r')

                new_labels = []
                while True:
                    line = labels_file_io.readline()

                    if not line:
                        break

                    line = line.split(' ')
                    label, center_x, center_y, width, height = (
                        line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    )

                    ratio_x = new_width / original_width
                    ratio_y = new_height / original_height

                    scaled_labels = [
                        label,
                        str(center_x * ratio_x),
                        str(center_y * ratio_y),
                        str(width * ratio_x),
                        str(height * ratio_y)
                    ]
                    new_labels.append(' '.join(scaled_labels))

                    # uncomment this to see the images with the new labels
                    # we suppose that the labels have the same name as the images (in yolo is the same)
                    # we multiply by the original width and height because we want to unnormalize the labels
                    #
                    # center_x = int(float(scaled_labels[1]) * original_width)
                    # center_y = int(float(scaled_labels[2]) * original_height)
                    # width = int(float(scaled_labels[3]) * original_width)
                    # height = int(float(scaled_labels[4]) * original_height)
                    # copy_image = image_resized.copy()
                    # cv2.rectangle(
                    #     copy_image,
                    #     (
                    #        int(center_x - width / 2),
                    #        int(center_y - height / 2)
                    #     ),
                    #     (
                    #         int(center_x + width / 2),
                    #         int(center_y + height / 2)
                    #     ),
                    #     (0, 255, 0),
                    #     2
                    # )
                    # cv2.imshow('image', copy_image)
                    # cv2.waitKey(10000) & 0xFF == ord('q')

                labels_file_io.close()

                if copy_in == 1:
                    labels_file_io = open(f'{label_directory_1}{labels_file}', 'w+')
                    labels_file_io.write('\n'.join(new_labels))
                    cv2.imwrite(f'{image_directory_1}{image_file}', image_resized)
                    labels_file_io.close()
                    continue

                labels_file_io = open(f'{label_directory_2}{labels_file}', 'w')
                labels_file_io.write('\n'.join(new_labels))
                cv2.imwrite(f'{image_directory_2}{image_file}', image_resized)
                labels_file_io.close()
