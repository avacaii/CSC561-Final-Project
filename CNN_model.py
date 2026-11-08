# Name: Ava Cai
# Date: 11/9/26
# Assignment: Project 3
# Purpose of the program: Set up CNN model and load dataset
# AI DECLARATION: AI was not used in the creation of this program.

# PyTorch: open source machine learning framework
import torch.nn as nn  # build and define neural networks
from torch.utils.data import DataLoader  # create custom datasets and batch data for training

# torchvision: computer vision tools built on top of PyTorch
from torchvision import models, transforms  # import pretrained CNN models and image preprocessing

# import dataset from MedMNIST: https://medmnist.com/
from medmnist import BloodMNIST


# Load pre-trained ResNet-50 CNN model with weights trained on ImageNet
# Learned this from: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
model = models.resnet50(weights="IMAGENET1K_V1")

# Set the model to evaluation mode
# This disables training-specific layers such as dropout and batch normalization updates 
# to ensuring consistent results during inference
# Learned this from: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
model.eval()

model.fc # Final fully connected layer that takes 2048 input features and outputs 1,000 classes
#print(model.fc) 

num_classes = 8 # The dataset has 8 classes 

# Create a sequential model with a dropout layer and a new linear layer
model.fc = nn.Sequential(
    nn.Dropout(0.5), #add dropout layer with a probability of 0.5
    nn.Linear(2048, num_classes) #add new layer with 2048 inputs and number of classes in our dataset as output size
)

for param in model.fc.parameters(): #unfreeze new model.fc layer's parameters
    param.requires_grad = True

preprocess = transforms.Compose([
    transforms.ToTensor(), #convert PIL Image to PyTorch tensor because the model is trained on tensors
    #I got the mean and standard deviation values for ImageNet from https://explore.albumentations.ai/transform/Normalize
    transforms.Normalize( #normalize pixel values using ImageNet mean and std
        mean=[0.485, 0.456, 0.406], #average RGB values from ImageNet
        std=[0.229, 0.224, 0.225], #standard deviations from ImageNet
    ),
])

# Load the dataset
# Dataset is BloodMNIST which contains images of 8 different types of bloodcells   
dataset = BloodMNIST(split="test", download=True)

# size=224 downloads the larger version of BloodMNIST images 
# The default is 28x28, which is too small for ResNet-50
# This size of image (224x224 pixels) matches ResNet's expected input size
# Use provided train-validation-test splits 
train_dataset = BloodMNIST(split="train", transform=preprocess, download=True, size=224)
val_dataset   = BloodMNIST(split="val",   transform=preprocess, download=True, size=224)
test_dataset  = BloodMNIST(split="test",  transform=preprocess, download=True, size=224)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)