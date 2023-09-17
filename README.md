# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project I developed code for an image classifier built with PyTorch, then convert it into a command line application.

# What does it do?

The project trains a model to classify different flowers for each type. It then makes a prediction of the flower by showing the accuracy

# What is needed?

conda env create -f environment.yaml
pip install -r requirement.txt

The model has been trained on [this dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

The folder flowers must be on the same folder of other files

# Usage

python train.py 

[-h] [--arch PRETRAINED_MODEL] [--save_dir SAVE_DIRECTORY] [--learning_rate LR] [--dropout DRPT]
                [--hidden_units UNITS] [--epochs NUM_EPOCHS] [--gpu]

python predict.py

 [-h] --image_dir IMAGE_DIR [--arch PRETRAINED_MODEL] [--load_dir CHECKPOINT] [--top_k TOPK]
                  [--category_names CATEGORY_NAMES] [--gpu]

# Example

python train.py flowers --arch vgg11 --save_dir checkpoint.pth --learning_rate 0.001 --dropout 0.2 --hidden_units 512 --epochs 4 --gpu

python.exe predict.py --image_dir flowers/test/30 --arch vgg11 --load_dir checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
