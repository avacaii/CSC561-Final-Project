# Name: Ava Cai
# Date: 11/9/26
# Assignment: Project 3
# Purpose of the program: Creates plots from training data

import json  # load the logistic regression history saved by log_reg_train.py
import torch  # load the saved checkpoint file
import matplotlib.pyplot as plt  # plot graphs and visualize training results

# Load the checkpoint of the trained model saved by train.py
checkpoint = torch.load("model.pth", map_location="cpu")
history = checkpoint["history"] # the training/validation loss and accuracy for each epoch

epochs = range(1, len(history["train_loss"]) + 1) #variable for the number of epochs based on the length of the trainig_loss in history

#plot training and validation loss
plt.figure()
plt.plot(epochs, history["train_loss"], label="Train Loss") #plot training loss over epochs (number of epochs as x-axis and training loss as y-axis)
plt.plot(epochs, history["val_loss"], label="Val Loss") #plot validation loss over epochs
plt.ylim(0, 1) #keep the y-axis range between 0 and 1 for clarity

#label graph and axes, include legend
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-50 Loss over Epochs")
plt.legend()
plt.grid(True)

#save the graph to Google Drive
plt.savefig("loss_curve.png")

#plot training and validation accuracy
plt.figure()
plt.plot(epochs, history["train_acc"], label="Train Acc") #plot training accuracy over epochs (number of epochs as x-axis and training accuracy as y-axis)
plt.plot(epochs, history["val_acc"], label="Val Acc") #plot validation accuracy over epochs
plt.ylim(0.5, 1) #keep the y-axis range between 0.5 and 1 for clarity

#label graph and axes, include legend
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet-50 Accuracy over Epochs")
plt.legend()
plt.grid(True)

#save the graph to Google Drive
plt.savefig("acc_curve.png")

# Load the logistic regression history saved by log_reg_train.py
with open("log_reg_history.json") as f:
    log_reg_history = json.load(f) # the training/validation loss and accuracy for each epoch

log_reg_epochs = range(1, len(log_reg_history["train_loss"]) + 1) #variable for the number of epochs based on the length of the training loss in log_reg_history

#plot logistic regression training and validation loss
plt.figure()
plt.plot(log_reg_epochs, log_reg_history["train_loss"], label="Train Loss") #plot training loss over epochs (number of epochs as x-axis and training loss as y-axis)
plt.plot(log_reg_epochs, log_reg_history["val_loss"], label="Val Loss") #plot validation loss over epochs
plt.ylim(0, 1.6) #keep the y-axis range between 0 and 1.6 because the logistic regression loss starts above 1

#label graph and axes, include legend
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Logistic Regression Loss over Epochs")
plt.legend()
plt.grid(True)

#save the graph to Google Drive
plt.savefig("log_reg_loss_curve.png")

#plot logistic regression training and validation accuracy
plt.figure()
plt.plot(log_reg_epochs, log_reg_history["train_acc"], label="Train Acc") #plot training accuracy over epochs (number of epochs as x-axis and training accuracy as y-axis)
plt.plot(log_reg_epochs, log_reg_history["val_acc"], label="Val Acc") #plot validation accuracy over epochs
plt.ylim(0.5, 1) #keep the y-axis range between 0.5 and 1 for clarity

#label graph and axes, include legend
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy over Epochs")
plt.legend()
plt.grid(True)

#save the graph to Google Drive
plt.savefig("log_reg_acc_curve.png")

#plot the validation accuracy of the CNN and logistic regression on the same graph to compare them
plt.figure()
plt.plot(epochs, history["val_acc"], label="ResNet-50 Val Acc") #plot CNN validation accuracy over epochs
plt.plot(log_reg_epochs, log_reg_history["val_acc"], label="Logistic Regression Val Acc") #plot logistic regression validation accuracy over epochs
plt.ylim(0.5, 1) #keep the y-axis range between 0.5 and 1 for clarity

#label graph and axes, include legend
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("ResNet-50 vs. Logistic Regression Validation Accuracy")
plt.legend()
plt.grid(True)

#save the graph to Google Drive
plt.savefig("comparison_acc_curve.png")

#show all the graphs at the same time, each in its own window
plt.show()
