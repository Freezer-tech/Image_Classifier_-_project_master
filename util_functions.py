import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image

from torchvision import datasets, transforms, models

# Function to load and preprocess the data
def load_data(data_dir):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define the data transforms for validation and testing data
    valid_test_transforms = transforms.Compose([
    transforms.Resize(256),            # Resize images to 256x256 pixels
    transforms.CenterCrop(224),        # Crop the center to 224x224 pixels
    transforms.ToTensor(),             # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize image data
    ])

    train_dataset = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
    valid_dataset = datasets.ImageFolder(data_dir + "/test", transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(data_dir + "/valid", transform=valid_test_transforms)

    # The trainloader will have shuffle=True so that the order of the images do not affect the model
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    return trainloader, testloader, validloader, train_dataset,valid_dataset, test_dataset

# Function to load and preprocess test image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Open the image using PIL
    # pil_image = Image.open(f'{image}' + '.jpg')
    pil_image = Image.open(f'{image}')
    
    # Define a series of image transformations
    image_transforms = transforms.Compose([
        transforms.Resize(256),             # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),         # Crop the center to 224x224 pixels
        transforms.ToTensor(),              # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],      # Normalize the image with specified mean and standard deviation
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Transforming image for use with network
    pil_to_netf = image_transforms(pil_image)
    
    # Converting to Numpy array 
    array_ima_tonetf = np.array(pil_to_netf)

    # Converting to torch tensor from Numpy array
    tensor_img = torch.from_numpy(array_ima_tonetf).type(torch.FloatTensor)

    # Adding dimension to image to comply with input of the model
    image_add_dim = tensor_img.unsqueeze_(0)

    return image_add_dim