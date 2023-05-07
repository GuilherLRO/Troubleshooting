import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
import logging
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Function to test the model on the test set
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    running_corrects = 0
    logger.info("Starting testing")
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        max_values, max_indices = torch.max(outputs, 1)
        preds = max_indices
        running_corrects += torch.sum(preds==labels.data).item()
    
    average_accuracy = running_corrects/len(test_loader.dataset)
    average_loss = test_loss/len(test_loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Accuracy: {100*average_accuracy}%')
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(average_loss, running_corrects, average_accuracy, 100.0 * running_corrects / len(test_loader.dataset)))


# Function to train the model
def train(model, train_loader, epochs, criterion, optimizer): 
    logger.info("Starting training")
    print("Starting training")
    for epoch in range(epochs):
        count = 0
        model.train()        
        # Loop over the batches in the train_loader
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            logger.info(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss.item()} | Images processed: {count}')
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * count,
                    count,
                    100.0 * batch_idx / count,
                    loss.item(),
                )
            )
                                                                                         
            if count > 500:
                break
    return model 
    

# Function to create data loaders for the train, test, and validation sets
def create_data_loaders(data_dir, batch_size):
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    validation_path = os.path.join(data_dir, 'valid')
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.Resize(256),
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader, validation_data_loader


# Main function to train and save the model
def main(args):
    print("começou")
    # Create the model
    model = models.resnet50(pretrained=True)
    # Freeze the weights of all layers in the model
    for param in model.parameters():
        param.required_grad = False 
    # Number of classes in the dataset -> 133 Breed classes
    num_classes = 133
    
    # Define the fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 256), 
                              nn.ReLU(),                 
                              nn.Linear(256, 128),
                              nn.ReLU(),
                              nn.Linear(128, num_classes),
                              nn.LogSoftmax(dim=1))

    # Define the loss function
    loss_criterion = nn.CrossEntropyLoss()
    # Define the optimizer to use for training
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    # Create data loaders for the training, validation, and test datasets
    train_data_loader, test_data_loader, _ = create_data_loaders(data_dir = args.data_dir, batch_size=args.batch_size)
    # Train the model for the specified number of epochs
    model = train(model, train_data_loader, args.epochs, loss_criterion, optimizer)
    # Test the model on the test dataset and print the results
    test(model, test_data_loader, loss_criterion)

    # Save the model
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Model saved")
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, metavar="N", help="input batch size for training")
    parser.add_argument( "--test_batch_size", type=int, default=1000, metavar="N", help="input batch size for testing")
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="training data path in S3")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="location to save the model to")
    
    args=parser.parse_args()
    
    main(args)