import argparse
import torch
import os
from PIL import Image
import json
from torch import nn
from torchvision import datasets, transforms, models
from util_functions import load_data, process_image
from functions import load_checkpoint, predict, test_model
import sys

def main():
    parser = argparse.ArgumentParser(description='Predict flower names using a trained deep learning model')

    parser.add_argument('--image_dir', action='store',required=True,
                        default=None,
                        help='Enter path to image. You can specify a folder or a single file')
    
    parser.add_argument('--arch', action='store',
                        dest='pretrained_model', default='vgg11',
                        help='Enter pretrained model to use; this classifier can currently work with\
                            VGG and Densenet architectures. The default is VGG-11.')

    parser.add_argument('--load_dir', action='store',
                        dest='checkpoint', default='checkpoint.pth',
                        help='Enter location to save checkpoint in.')

    parser.add_argument('--top_k', action='store',
                        dest='topk', type=int, default=3,
                        help='Enter number of top most likely classes to view, default is 3.')

    parser.add_argument('--category_names', action='store',
                        dest='category_names', default='cat_to_name.json',
                        help='Enter path to category names mapping file (json file).')

    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Turn GPU mode on or off, default is off.')

    args = parser.parse_args()

    image_dir = args.image_dir
    load_dir = args.checkpoint
    top_k = args.topk
    category_names = args.category_names
    gpu = args.gpu

    # Upload category mapping
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Establish model template
    pre_tr_model = args.pretrained_model
    model = getattr(models, pre_tr_model)(pretrained=True)

    # Check if checkpoint exists
    if os.path.isfile(load_dir):
        try:
            # Load checkpoint
            loaded_checkpoint = load_checkpoint(model, load_dir, gpu)

            # Check whether the checkpoint weights match the current model
            if checkpoint_weights_match(loaded_checkpoint, model):
                model = loaded_checkpoint
                print("The model has been loaded from the checkpoint.\n")
            else:
                print("The checkpoint weights do not match the current model. Loading a new model.\n")
        except FileNotFoundError:
            print(f"Error: The file '{load_dir}' does not exist. Make sure to specify the correct file path.\n")
            return

    # Get a list of all image files in the directory
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_path in image_files:
        # Preprocess each image
        processed_image = process_image(image_path)

        # Run the prediction
        probs, classes = predict(processed_image, model, top_k, gpu)

        # Print results for each image
        print(f"\nResults for {image_path}:\n")
        print(probs)
        print(classes)
        for i in range(top_k):
            # Get class names
            class_names = [cat_to_name[class_idx] for class_idx in classes]
            # Print results
            print(f"This flower should be: {class_names[i]}, with probability of: {probs[i]*100:.4f}%")            

    # Check whether the ESC key or CTRL+C was pressed.
    try:
        while True:
            user_input = input("Press ESC or CTRL+C to exit: ")
            if user_input == '\x1b':  # Esc key
                break
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)

def checkpoint_weights_match(checkpoint, model):
    # Check whether the checkpoint weights match the current model
    checkpoint_state_dict = checkpoint.state_dict()
    model_state_dict = model.state_dict()

    for key in checkpoint_state_dict:
        if key in model_state_dict:
            if checkpoint_state_dict[key].shape != model_state_dict[key].shape:
                return False
        else:
            return False

    return True

if __name__ == '__main__':
    main()