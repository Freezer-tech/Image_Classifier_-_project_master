import argparse
import torch
from torch import __version__
from torch import nn, optim
from torchvision import datasets, transforms, models
from util_functions import load_data
from functions import build_classifier, validation, train_model, test_model, save_model, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a flower dataset')

    parser.add_argument('data_directory', action='store',
                        help='Enter path to training data.')

    parser.add_argument('--arch', action='store',
                        dest='pretrained_model', default='vgg11',
                        help='Enter pretrained model to use; this classifier can currently work with\
                            VGG and Densenet architectures. The default is VGG-11.')

    parser.add_argument('--save_dir', action='store',
                        dest='save_directory', default='checkpoint.pth',
                        help='Enter location to save checkpoint in.')

    parser.add_argument('--learning_rate', action='store',
                        dest='lr', type=float, default=0.001,
                        help='Enter learning rate for training the model.')

    parser.add_argument('--dropout', action='store',
                        dest='drpt', type=float, default=0.05,
                        help='Enter dropout for training the model, default is 0.05.')

    parser.add_argument('--hidden_units', action='store',
                        dest='units', type=int, default=500,
                        help='Enter number of hidden units in classifier, default is 500.')

    parser.add_argument('--epochs', action='store',
                        dest='num_epochs', type=int, default=2,
                        help='Enter number of epochs to use during training, default is 2.')

    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Turn GPU mode on or off, default is off.')

    args = parser.parse_args()

    data_dir = args.data_directory
    save_dir = args.save_directory
    arch = args.pretrained_model
    learning_rate = args.lr
    dropout = args.drpt
    hidden_units = args.units
    epochs = args.num_epochs
    gpu_mode = args.gpu

    # Load and preprocess data 
    trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)
    
    # Upload pretrained model
    model = getattr(models, arch)(pretrained=True)

    # Build and attach new classifier
    input_units = model.classifier[0].in_features
    model = build_classifier(model, input_units, hidden_units, dropout)

    # Recommended to use NLLLoss when using Softmax
    criterion = nn.NLLLoss()
    # Using Adam optimiser which makes use of momentum to avoid local minima
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the model
    model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode)

    # Test the model
    test_model(model, testloader, gpu_mode)

    # Save the model
    save_model(model, train_data, optimizer, save_dir, epochs)

if __name__ == '__main__':
    main()