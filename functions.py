import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to build new classifier
def build_classifier(model, input_units, hidden_units, dropout):
    # Freeze the pre-trained model's parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier part of the model
    classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units,102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

# Function for the validation pass
def validation(model, validloader, criterion, gpu_mode):
    model.eval()
    correct = 0 # torch.FloatTensor([0]).to(device)
    total = 0 # torch.FloatTensor([0]).to(device)
    valid_loss = 0.0
    
    with torch.no_grad():             
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device) # Move inputs and labels to the appropriate device (CPU or GPU)
            outputs = model(inputs)                               # Forward pass to get model predictions
            loss = criterion(outputs, labels)                     # Calculate the loss between the model's predictions and the actual labels
            valid_loss += loss.item()                             

            probabilities = torch.exp(outputs)                    # Calculate probabilities and predicted labels
            _, predicted = torch.max(probabilities, 1)            
            correct += (predicted == labels).sum().float()        # Check correct predictions and count total labels
            total += labels.size(0)                               
            
    valid_loss = valid_loss / len(validloader)
    valid_accuracy = 100 * correct / total

    return valid_loss, valid_accuracy

# Function to train the model
def train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu_mode):
    # Number of epochs
    # num_epochs = 4
    model.to(device)  # Transfer the model to CPU or GPU

    for epoch in range(epochs):
        # since = time.time()
        running_loss = 0
        #since = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
    
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)     # Move inputs and labels to device
            optimizer.zero_grad()                                     # Zero out previous gradients 
            outputs = model(inputs)                                   # Forward pass, compute loss, and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()                                          # Update model parameters
            train_loss += loss.item()                                 # Accumulate training loss
        
            # Calculate accuracy during training
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        # Calculate and print the average training loss and accuracy for this epoch
        train_loss = train_loss / len(trainloader)
        train_accuracy = 100 * correct / total

        # Validate the model at the end of each epoch
        valid_loss, valid_accuracy = validation(model, validloader, criterion, gpu_mode)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {float(train_accuracy):.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {float(valid_accuracy):.2f}%")

    #time_taken = time.time() - since
    #print(f"Time taken for epoch: {(time_taken):.2f} seconds")
    # Turning training back on
    model.train()
    
    return model, optimizer

# Function to test the model
def test_model(model, testloader, gpu_mode):
    correct = 0
    total = 0

    model.to(device)  # Transfer the model to CPU or GPU
    with torch.no_grad():             
        for data in testloader:
            images, labels = data 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                               
        
            _, predicted_out = torch.max(outputs.data, 1)            
            total += labels.size(0)                               
            correct += (predicted_out == labels).sum().item()
            
    print(f"Test accuracy of model: {round(100 * correct / total,3)}%")

# Function to save the model in checkpoint.pth
def save_model(model, train_data, optimizer, save_dir, epochs):
    # Saving: feature weights, new classifier, index-to-class mapping, optimiser state, and No. of epochs

    checkpoint = {'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': train_data.class_to_idx,
              'opt_state': optimizer.state_dict(),
              'epochs': epochs}

    torch.save(checkpoint, save_dir) 

# Function to load the model from file checkpoint.pth
def load_checkpoint(model, save_dir, gpu_mode):

    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
        model.to('cuda')
    else:
        checkpoint = torch.load(save_dir, map_location='cpu')
        model.to('cpu')
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Function to predict the model
def predict(processed_image, loaded_model, topk, gpu_mode):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Transfer the processed image to the GPU
    if gpu_mode == True:
        processed_image = processed_image.to('cuda')
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()

    # Set the model to evaluation mode
    loaded_model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = loaded_model(processed_image)
        
    # Calculating probabilities
    probabilities = torch.exp(outputs)
    top_probabilities = probabilities.topk(topk)[0]
    top_indices = probabilities.topk(topk)[1]
    
    # Invert the class_to_idx dictionary to get idx_to_class
    idx_to_class = {v: k for k, v in loaded_model.class_to_idx.items()}
    
    # Map the indices to class labels
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    return top_probabilities[0].tolist(), top_classes