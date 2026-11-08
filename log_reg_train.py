# Name: Ava Cai
# Date: 11/7/26
# Assignment: Project 3
# Purpose of the program: Train a logistic regression baseline on raw BloodMNIST pixels

import json  # save the training history to a file that plot.py can read
import numpy as np  # shuffle the training data each epoch
from medmnist import BloodMNIST  # import dataset from MedMNIST: https://medmnist.com/
from sklearn.linear_model import SGDClassifier  # linear classifier trained with stochastic gradient descent
from sklearn.metrics import log_loss  # cross-entropy loss, the same loss the CNN uses
from sklearn.preprocessing import StandardScaler  # rescales each feature to mean 0 and standard deviation 1

num_classes = 8  # The dataset has 8 classes


def load(split):
    """
    Loads one split of BloodMNIST and flattens each image into a feature vector.

    Logistic regression expects each example to be a 1D row of numbers, so each
    28x28 RGB image is flattened into 28 * 28 * 3 = 2352 pixel features.

    Args:
        split (str): Which split to load ('train', 'val', or 'test').

    Returns:
        tuple containing:
        - X (numpy.ndarray): Pixel features with shape (N, 2352), scaled to [0, 1].
        - y (numpy.ndarray): Class labels (0-7) with shape (N,).
    """
    # size=28 loads the small 28x28 version of the images
    # 224x224 images would give 150,528 features per image, which is far too many for logistic regression
    ds = BloodMNIST(split=split, download=True, size=28)

    # ds.imgs is a NumPy array of shape (N, 28, 28, 3)
    # reshape(len(ds.imgs), -1) keeps one row per image and flattens the rest into 2352 columns
    # dividing by 255 scales pixel values from [0, 255] down to [0, 1]
    X = ds.imgs.reshape(len(ds.imgs), -1) / 255.0

    # BloodMNIST labels have shape (N, 1), ravel() flattens them to shape (N,) which scikit-learn expects
    y = ds.labels.ravel()

    return X, y


def evaluate(clf, X, y):
    """
    Computes the cross-entropy loss and accuracy of the logistic regression model.

    Args:
        clf (sklearn.linear_model.SGDClassifier): The logistic regression model being evaluated.
        X (numpy.ndarray): Standardized pixel features with shape (N, 2352).
        y (numpy.ndarray): Class labels (0-7) with shape (N,).

    Returns:
        tuple containing:
        - loss (float): The average cross-entropy loss over all samples.
        - accuracy (float): The accuracy of the model (correct predictions / total samples).
    """
    probs = clf.predict_proba(X)  # predicted probability of each of the 8 classes, shape (N, 8)
    # labels=range(num_classes) tells log_loss which column belongs to which class
    loss = log_loss(y, probs, labels=range(num_classes))
    accuracy = clf.score(X, y)  # score() returns the accuracy (correct predictions / total images)
    return loss, accuracy


def train_log_reg(X_train, y_train, X_val, y_val, num_epochs=30, seed=0):
    """
    Trains a multinomial logistic regression model one epoch at a time.

    scikit-learn's LogisticRegression trains in a single call with no epochs, so this
    uses SGDClassifier with log loss instead, which is logistic regression trained with
    stochastic gradient descent. Calling partial_fit() makes one pass over the training
    data, so each call is one epoch, the same as the CNN in train.py. This lets the
    loss and accuracy curves of the two models be compared epoch by epoch.

    Args:
        X_train (numpy.ndarray): Standardized training features with shape (N, 2352).
        y_train (numpy.ndarray): Training labels with shape (N,).
        X_val (numpy.ndarray): Standardized validation features with shape (M, 2352).
        y_val (numpy.ndarray): Validation labels with shape (M,).
        num_epochs (int, optional): Number of epochs to train. Defaults to 30 to match the CNN.
        seed (int, optional): Random seed so the results are reproducible. Defaults to 0.

    Returns:
        dict: A dictionary containing the training and validation history with keys:
        - "train_loss" (list): Training loss after each epoch.
        - "train_acc" (list): Training accuracy after each epoch.
        - "val_loss" (list): Validation loss after each epoch.
        - "val_acc" (list): Validation accuracy after each epoch.
    """
    # loss="log_loss" makes SGDClassifier a logistic regression model
    # with more than 2 classes, predict_proba() normalizes the per-class scores into probabilities that sum to 1
    # learning_rate="constant" with eta0=1e-4 keeps the step size fixed at 1e-4, the same learning rate as the CNN
    # The default step size schedule takes very large early steps, which made the weights blow up and
    # gave overconfident predictions with a cross-entropy loss above 5
    clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=1e-4, random_state=seed)
    rng = np.random.default_rng(seed)  # random number generator used to shuffle the data

    # store training and validation history for plotting later (same keys as train.py)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):  # loop over all epochs
        # shuffle the training data each epoch, the same as shuffle=True in the CNN's DataLoader
        order = rng.permutation(len(X_train))

        # one pass of gradient descent over the training data
        # classes must be given because partial_fit() never sees the whole dataset at once
        clf.partial_fit(X_train[order], y_train[order], classes=np.arange(num_classes))

        # measure loss and accuracy on the training and validation data after the epoch
        # (the CNN averages its training metrics during the epoch, this measures them at the end)
        train_loss, train_acc = evaluate(clf, X_train, y_train)
        val_loss, val_acc = evaluate(clf, X_val, y_val)

        # save to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # print progress summary for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history


if __name__ == "__main__":
    # Load the provided train and validation splits
    # (the test set is saved for the very end, the same as the CNN in train.py)
    X_train, y_train = load("train")
    X_val, y_val = load("val")

    # Standardize the features so every pixel has mean 0 and standard deviation 1
    # Gradient descent converges much faster when all features are on the same scale
    # The scaler is fit on the training data only so no information from the validation set leaks into training
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Train the model and save the loss/accuracy for each epoch
    history = train_log_reg(X_train, y_train, X_val, y_val)

    # Save the history into the current folder so plot.py can graph it
    with open("log_reg_history.json", "w") as f:
        json.dump(history, f)
