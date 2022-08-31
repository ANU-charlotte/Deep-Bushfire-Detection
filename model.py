import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from GNN import GNNNet
import torch
from GNN import GNNNet
'''
model.py defines the Faster RCNN pre-trained model which is already built-in on torch
'''

def create_model(num_classes):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Number of input features
    in_features = model.roi_heads.box_predictor
    return in_features, model    # Extract feature

def generate_prediction_with_updated_features(model,updated_feature,num_classes):
    bnd_box_feat = updated_feature.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(bnd_box_feat,num_classes)
    return model



