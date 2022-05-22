import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
'''
model.py defines the Faster RCNN pre-trained model which is already built-in on torch
'''
def create_model(num_classes):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model