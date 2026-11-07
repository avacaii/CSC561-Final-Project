# Name: Ava Cai
# Date: 11/9/26
# Assignment: Project 3
# Purpose of the program: Train and evaluate functions
# AI DECLARATION: AI was not used in the creation of this program.

import torch  # main PyTorch library
import torch.nn as nn  # build and define neural networks (used for the loss function)
from torch import optim  # optimization algorithms for training
from tqdm import tqdm  # show progress bars during training loops


def get_device():
    """
    Picks the best available device to run the model on.

    Returns:
        str: 'cuda' for an NVIDIA GPU (e.g. a T4), 'mps' for an Apple Silicon Mac GPU,
        or 'cpu' if neither is available.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, loss, device):

    """
    Evaluates a trained PyTorch model on the validation dataset.

    This function disables gradient computation to improve efficiency, then iterates
    through all batches in the provided DataLoader to compute the model's average loss
    and accuracy. It measures how well the model generalizes to unseen data.

    Args:
        model (torch.nn.Module): The trained PyTorch model to be evaluated.
        loader (torch.utils.data.DataLoader): DataLoader providing the validation or test dataset.
        loss (torch.nn.Module): The loss function used to measure prediction error.
        device (str): The device to perform computation on ('cpu' or 'cuda').

    Returns:
        tuple containing:
        - avg_loss (float): The average loss value over all samples in dataset.
        - accuracy (float): The overall accuracy of the model (correct predictions / total samples).

    """

    model.eval() #sets model to evaluation mode
    
    # Initializes variables for the value of the loss, the number of images the model correctly predicts, 
    # and the total number of images the model is tested on
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # Disable gradient calculations to save memory and computation
        # Loop through testing batches
        for images, labels in loader: 
            # Move data to the device being used to train the model (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            # BloodMNIST labels have shape [batch, 1], flatten them to [batch] for loss function and accuracy check 
            labels = labels.squeeze(1) 

            outputs = model(images) #forward pass to get the model's predictions
            loss_val = loss(outputs, labels) #compute loss for batch by comparing predictions with actual class labels
            val_loss += loss_val.item() * images.size(0) #accumulate the total loss

            _, predicted = torch.max(outputs, 1) #get class with highest predicted probability 
            val_correct += (predicted == labels).sum().item() #count correct predictions
            val_total += labels.size(0) #count total images the model was tested on

    #compute average loss and overall accuracy
    avg_loss = val_loss / val_total #average loss for the testing data
    accuracy = val_correct / val_total #accuracy of the model

    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-4, device=None):

    """
    Trains a PyTorch model using the provided training and validation datasets.

    This function handles the full training process, including forward passes,
    loss computation, backpropagation, and parameter updates. After each epoch,
    it also evaluates the model on the validation dataset to monitor generalization
    performance. Training and validation metrics are stored in history.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 30.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        device (str, optional): Device to run training on ('cuda', 'mps', or 'cpu').
            Defaults to None, which picks the best available device automatically.

    Returns:
        dict: A dictionary containing the training and validation history with keys:
        - "train_loss" (list): Average training loss for each epoch.
        - "train_acc" (list): Training accuracy for each epoch.
        - "val_loss" (list): Validation loss for each epoch.
        - "val_acc" (list): Validation accuracy for each epoch.

    """

    # If no device was given, pick the best one available (GPU if possible, otherwise CPU)
    # The GPU name depends on the computer, on a Mac with Apple Silicon it is "mps",
    # on Windows/Linux with an NVIDIA GPU it is "cuda", and with no GPU it falls back to "cpu"
    if device is None:
        device = get_device()

    model.to(device) #move model to GPU if it is available
    criterion = nn.CrossEntropyLoss() #define the loss function, CrossEntropyLoss is commonly used for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #the Adam optimizer is used for adaptive learning rate

    #store training and validation history for plotting later
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs): #loop over all epochs
        model.train() #set the model to training mode which enables dropout and batchnorm updates
        #initialize variables for the loss, correct predictions, and total number of images the model is being trained on
        running_loss = 0.0
        correct = 0
        total = 0

        #training loop
        for images, labels in tqdm(train_loader): #iterate through batches
            images, labels = images.to(device), labels.to(device) #move data to the device being used (GPU or CPU)
            labels = labels.squeeze(1) #BloodMNIST labels have shape [batch, 1], so flatten them to [batch] for the loss function and accuracy check

            optimizer.zero_grad() #reset gradients from previous step
            outputs = model(images) #forward pass to get the model's predictions
            loss = criterion(outputs, labels) #compute loss using nn.CrossEntropyLoss()
            loss.backward() #backpropagate to compute gradients
            optimizer.step() #update model parameters (weights) using the gradients

            running_loss += loss.item() * images.size(0) #track the loss of the epoch
            _, predicted = torch.max(outputs, 1) #get class with highest predicted probability
            correct += (predicted == labels).sum().item() #count correct predictions
            total += labels.size(0) #count total images the model was trained on

        #compute the average loss and accuracy for the epoch
        train_loss = running_loss / total
        train_acc = correct / total

        #evaluate on validation data after each epoch (the test set is saved for the very end)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        #save to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        #print progress summary for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history
