import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib
from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

# Specify command line options for running on cmd
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video',
    default='Deep Smoke Detection.v1i.voc (1)/video/testing.mp4')
args = vars(parser.parse_args())

# Load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

detection_threshold = 0.8
RESIZE_TO = (1920, 1080)
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

save_name = str('videoTest')
# Create VideoWriter object
out = cv2.VideoWriter(f"inference_outputs/videos/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      RESIZE_TO)

# Read entire video
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, RESIZE_TO)
        image = frame.copy()
        image = image.astype(np.float32)
        # Normalise pixel range
        image /= 255.0
        # TODO: Removed line for BRG to RGB conversion, add to see difference
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) # Might try removing this line too...
        # To tensor
        image = torch.tensor(image, dtype=torch.float)
        # Add batch dimension
        image = torch.unsqueeze(image, 0)
        # Get the start time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Only continue if there are detected 'Smoke' objects
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # Filter out boxes according to detection_threshold
            # TODO: might try increasing threshold...
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # Draw bounding boxes
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = (0, 255, 0)
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color, 2)
                # Adding name to object
                cv2.putText(frame, class_name,
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color,
                            1, lineType=cv2.LINE_AA)

        cv2.imshow('image', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
