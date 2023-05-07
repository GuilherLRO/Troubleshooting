import os
import sys
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile

import smdebug.pytorch as smd


# Allow truncated images to be loaded without throwing errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, hook):
    """Test the model on a test dataset."""
    model.eval()  # Set the model to evaluation mode
    hook.set_mode(smd.modes.EsmdVAL)  # Set the mode for the debugger hook
    test_loss = 0
    running_corrects = 0

    # Iterate over the test dataset
    for inputs, labels in test_loader:
        # Forward pass
        outputs = model(inputs)
        # Calculate the loss
        test_loss += criterion(outputs, labels).item()
        # Get the predicted labels
        _, preds = torch.max(outputs, 1)
        # Count the number of correct predictions
        running_corrects += torch.sum(preds == labels.data).item()

    # Calculate the average accuracy and loss
    average_accuracy = running_corrects / len(test_loader.dataset)
    average_loss = test_loss / len(test_loader.dataset)
    # Log the results
    logger.info(f'Test set: Average loss: {average_loss}, Average accuracy: {100 * average_accuracy}%')


def train(model, train_loader, validation_loader, epochs, criterion, optimizer, hook):
    """Train the model on a training dataset."""
    # count = 0
    # Iterate over the epochs
    for epoch in range(epochs):
        # Set the mode for the debugger hook
        hook.set_mode(smd.modes.TRAIN)
        # Set the model to training mode
        model.train()

        # Iterate over the training dataset
        for inputs, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Count the number of training examples
            count += len(inputs)
            # if count > 500:
            #     break

        # Set the mode for the debugger hook
        hook.set_mode(smd.modes.EVAL)
        # Set the model to evaluation mode
        model.eval()
        running_corrects = 0

        # Iterate over the validation dataset
        with torch.no_grad():
            for inputs, labels in validation_loader:
                # Forward pass
                outputs = model(inputs)
                # Calculate the loss
                loss = criterion(outputs, labels)
                # Get the predicted labels
                _, preds = torch.max(outputs, 1)
                # Count the number of correct predictions
                running_corrects += torch.sum(preds == labels.data).item()

        # Calculate the total accuracy
        total_accuracy = running_corrects / len(validation_loader.dataset)
        # Log the results
        logger.info(f'Validation set: Average accuracy: {100 * total_accuracy}%')
    # Return the trained model
    return model

def net():
    # Load a pretrained ResNet-50 model
    model = models.resnet50(pretrained=True)
    
    # Freeze all of the model's parameters except for the fully connected layer's parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer with a new one that produces the desired number of output classes
    num_features = model.fc.in_features
    num_classes = 133
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    return model

def create_data_loaders(data, batch_size):
    # set up paths to training, testing, and validation data
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    
    # define image transforms for data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # create datasets and data loaders for training, testing, and validation data
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    # check if GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # create model and move it to device
    model = net()
    model = model.to(device)
    
    # define loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # set up SageMaker Debugger hook for real-time model monitoring
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    # create data loaders for training, testing, and validation data
    train_data_loader, test_data_loader, validation_data_loader = create_data_loaders(data=args.data_dir, batch_size=args.batch_size)
    
    # train the model and save the best weights based on validation loss
    model = train(model, train_data_loader, validation_data_loader, args.epochs, loss_criterion, optimizer, hook)
    
    # test the model on the test data and log performance metrics
    test(model, test_data_loader, loss_criterion, hook)
    
    # save the model
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Model saved")


if __name__=='__main__':
     # parse command-line arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, metavar="N", help="input batch size for training")
    parser.add_argument( "--test_batch_size", type=int, default=1000, metavar="N", help="input batch size for testing")
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="training data path in S3")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="location to save the model to")
    args=parser.parse_args()
    
    main(args)