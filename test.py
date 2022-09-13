import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model import create_model,generate_prediction_with_updated_features
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
import time
import torch
import torch.nn as nn
from torchvision import models
'''
test.py runs model on testing images, and display the images. Similar to video.py.
'''

# load the best model and trained weights
in_features, model, num_classes = create_model(num_classes=NUM_CLASSES)

# Introduce GNN after in_feature

model = generate_prediction_with_updated_features(model, in_features, num_classes)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(model.to(DEVICE).eval())

# directory where all the images are present
DIR_TEST = 'data/testing'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")
# classes: 0 index is reserved for background
CLASSES = [
    'background', 'Smoke'
]

detection_threshold = 0.8

for i in range(len(test_images)):
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    cv2.imshow('test',image)
    orig_image = image.copy()
    image = image.astype(np.float32)
    # Normalise
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)


    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # Draw bounding boxes
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, lineType=cv2.LINE_AA)
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(0)
        cv2.imwrite(f"test_predictions/{image_name}.jpg", orig_image, )
    print(f"Image {i + 1} done...")
    print('-' * 50)
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()