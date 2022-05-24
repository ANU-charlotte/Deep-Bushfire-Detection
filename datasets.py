"""
datasets.py includes custom data loader, custom dataset class
"""
import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform

class MyDataset(Dataset):
    """
    Custom dataset class
    """
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms # No transformations
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        """
        Returns the data of [idx] indexed image
        """
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        image = cv2.imread(image_path)
        image = image.astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0  # Normalise image

        # Extract XML file for getting the bounding-box annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        image_height, image_width = image.shape

        # Box coordinates
        for member in root.findall('object'):
            # Returns all 'Smoke' object names and return their index
            labels.append(self.classes.index(member.find('name').text))

            x_min = int(member.find('bndbox').find('xmin').text)
            x_max = int(member.find('bndbox').find('xmax').text)
            y_min = int(member.find('bndbox').find('ymin').text)
            y_max = int(member.find('bndbox').find('ymax').text)

            x_min = (x_min / image_width) * self.width
            x_max = (x_max / image_width) * self.width
            y_min = (y_min / image_height) * self.height
            y_max = (y_max / image_height) * self.height

            boxes.append([x_min, y_min, x_max, y_max])

        # To tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area
        w = (boxes[:, 2] - boxes[:, 0])
        h = (boxes[:, 3] - boxes[:, 1])
        area = w*h
        # Ensure no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Prepare `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def create_train_dataset():
    """
    Create Training Dataset
    """
    train_dataset = MyDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    return train_dataset


def create_valid_dataset():
    """
        Create Validation Dataset
    """
    valid_dataset = MyDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    """
        Create Training Loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    """
        Create Validation Loader
    """
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader



if __name__ == '__main__':
    dataset = MyDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )


    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)