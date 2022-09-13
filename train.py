"""
train.py performs model training and validation
"""
from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS,
)
from model import create_model, generate_prediction_with_updated_features
from custom_utils import AverageLoss, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)
import argparse
import torch
import matplotlib.pyplot as plt
import time
import ssl
from GNN import GNNNet
plt.style.use('ggplot')
# Allow download of fasterRCNN ResNet 50
ssl._create_default_https_context = ssl._create_unverified_context # Can comment out if there is no issue while downloading

def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list

    # Initialize tqdm progress bar
    progressBar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(progressBar):
        optimizer.zero_grad()   # Zero gradient optimizer
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)

        # Backwards Loss propagation
        losses.backward()
        optimizer.step()
        train_itr += 1

        # Update loss each iteration
        progressBar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# Validating validation set
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    progressBar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(progressBar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        progressBar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

# GNN codes

if __name__ == '__main__':
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")
    # Initialise the model and move to the computation device
    in_features, model, num_classes = create_model(num_classes=2)
    print(in_features.size())
    model = generate_prediction_with_updated_features(model, in_features, num_classes)
    DEVICE = torch.device('cpu')
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    del checkpoint['model_state_dict']['roi_heads.box_predictor.cls_score.weight']
    del checkpoint['model_state_dict']['roi_heads.box_predictor.cls_score.bias']
    del checkpoint['model_state_dict']['roi_heads.box_predictor.bbox_pred.weight']
    del checkpoint['model_state_dict']['roi_heads.box_predictor.bbox_pred.bias']

    # Freeze
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    freeze = ['FastRCNNPredictor']  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    torch.save(checkpoint, "modified_model.pth")
    new_checkpoint = torch.load("modified_model.pth", map_location=DEVICE)

    print(model)
    GNNmodel = GNNNet(infeature)
    new_params = GNNmodel.state_dict().copy()
    GNNmodel.load_state_dict(new_params)
    updated_feature = GNNmodel
    model = generate_prediction_with_updated_features(model, updated_feature, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # Get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # Initialize the AverageLoss class
    train_loss_hist = AverageLoss()
    val_loss_hist = AverageLoss()
    train_itr = 1
    val_itr = 1
    # List for storing loss values
    train_loss_list = []
    val_loss_list = []
    MODEL_NAME = 'model'
    # Display images prior to training
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image

        show_tranformed_image(train_loader)
    # Initialize SaveBestModel class
    save_best_model = SaveBestModel()
    # Start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")
        # Reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # Start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # Save the best model
        save_best_model(
            val_loss_hist.value, epoch, model, GNNmodel, optimizer
        )
        # Save current epoch model
        save_model(epoch, model, GNNmodel, optimizer)
        # Save loss plot
        save_loss_plot(OUT_DIR, train_loss, val_loss)

