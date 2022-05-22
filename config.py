import torch

BATCH_SIZE = 8
RESIZE_TO = 416
NUM_EPOCHS = 10
NUM_WORKERS = 4

#DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')
# Training and Validation directory
TRAIN_DIR = 'Deep Smoke Detection.v1i.voc (1)/train'
VALID_DIR = 'Deep Smoke Detection.v1i.voc (1)/valid'

CLASSES = [
    '__background__', 'Smoke'
]
NUM_CLASSES = len(CLASSES)

# Display images
VISUALIZE_TRANSFORMED_IMAGES = True

# Location to save model and plots
OUT_DIR = 'outputs'