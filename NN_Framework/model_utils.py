import torch
import torch.nn as nn
import tifffile as tiff
import h5py
import os
import datetime
import pathlib as Path
import numpy as np
import polars as pl
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
import mlflow
#TODO switch to logging after initial tests


class EarlyStopping:
    """EarlyStopping is a utility class to stop training when a monitored metric has stopped improving.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): If True, prints a message for each validation loss improvement.
        path (str): Path to save the model when the validation loss improves.
        counter (int): Counter for the number of epochs since the last improvement.
        best_loss (float): The best validation loss observed.
        early_stop (bool): Flag indicating whether to stop training.
    """
    def __init__(self, patience=5, delta=0.01, verbose=True, path='best_model.pth'):
        """Initializes the EarlyStopping object.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path to save the model when the validation loss improves.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        """Checks if the validation loss has improved and updates the early stopping criteria.

        Args:
            val_loss (float): The current validation loss.
            model (nn.Module): The model to save if the validation loss improves.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Saves the model when the validation loss improves.

        Args:
            model (nn.Module): The model to save.
        """
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Model saved with improved validation loss: {self.best_loss:.4f}")


def compute_metrics(predictions, labels, num_classes, device):
    """
    Compute metrics for model evaluation.

    This function calculates various performance metrics including accuracy, F1 score, precision, recall, sensitivity, and specificity
    based on the model predictions and true labels.

    Args:
        predictions (torch.Tensor): The predicted class labels from the model.
        labels (torch.Tensor): The true class labels.
        num_classes (int): The number of classes in the classification task.
        device (torch.device): The device to perform the computations on (CPU or GPU).

    Returns:
        dict: A dictionary containing the computed metrics:
            - accuracy (float): The accuracy of the model.
            - f1_score (float): The F1 score of the model.
            - precision (float): The precision of the model.
            - recall (float): The recall of the model.
            - sensitivity (list): The sensitivity for each class.
            - specificity (list): The specificity for each class.
    """
    if num_classes==2:
        task_type='binary'
    else:
        task_type='multiclass'
    accuracy = Accuracy(task=task_type,num_classes=num_classes, average='macro').to(device)
    f1 = F1Score(task=task_type,num_classes=num_classes, average='macro').to(device)
    precision = Precision(task=task_type,num_classes=num_classes, average='macro').to(device)
    recall = Recall(task=task_type,num_classes=num_classes, average='macro').to(device)
    confusion_matrix = ConfusionMatrix(task=task_type,num_classes=num_classes).to(device)

    acc = accuracy(predictions, labels) * 100
    f1_score = f1(predictions, labels) * 100
    prec = precision(predictions, labels) * 100
    rec = recall(predictions, labels) * 100
    cm = confusion_matrix(predictions, labels)

    sensitivity = cm.diag() / cm.sum(1) * 100  # TP / (TP + FN)
    specificity = (cm.sum() - cm.sum(1) - cm.sum(0) + cm.diag()) / (cm.sum() - cm.sum(1)) * 100  # TN / (TN + FP)

    return {
        "accuracy": acc.item(),
        "f1_score": f1_score.item(),
        "precision": prec.item(),
        "recall": rec.item(),
        "sensitivity": sensitivity.cpu().numpy().tolist(),
        "specificity": specificity.cpu().numpy().tolist(),
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, location, 
    epochs=50, patience=25, delta=0.00000001, check_val_freq=5,num_classes=2, log_with_mlflow=True,mlflow_uri="http://127.0.0.1:5000"):
    """
    Train Model

    This function trains a given model using the provided training and validation data loaders. It utilizes a specified loss function and optimizer, and implements early stopping based on validation loss.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (callable): The loss function to be used.
        optimizer (Optimizer): The optimizer for updating model weights.
        device (torch.device): The device to perform computations on (CPU or GPU).
        location (str): The directory to save the best model.
        epochs (int, optional): The number of epochs to train the model. Default is 50.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Default is 25.
        delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Default is 1e-8.
        check_val_freq (int, optional): Frequency of validation checks. Default is 5.
        num_classes (int, optional): The number of classes in the classification task. Default is 2.
        log_with_mlflow (bool, optional): Whether to log training with MLflow. Default is True.
        mlflow_uri (str, optional): The URI for the MLflow tracking server. Default is "http://127.0.0.1:5000".

    Returns:
        None
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, delta=delta, path=os.path.join(location, 'best_model.pth'))
    writer = SummaryWriter(log_dir=location)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    if log_with_mlflow:
        mlflow.set_tracking_uri(mlflow_uri)  
        mlflow.start_run()  

        # Log model and optimizer parameters
        for param in ["img_size_x", "patch_size_x", "embed_dim", "num_heads", "depth", "mlp_dim", "dropout_rate", "weight_decay"]:
            mlflow.log_param(param, model.hparams.get(param, float('nan')))

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        all_train_predictions, all_train_labels = [], []

        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
           # print(labels,predicted,loss)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            all_train_predictions.append(predicted)
            all_train_labels.append(labels)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        train_metrics = compute_metrics(
            torch.cat(all_train_predictions), torch.cat(all_train_labels), num_classes, device
        )

        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)

        if log_with_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric("train_f1_score", train_metrics["f1_score"], step=epoch)
            mlflow.log_metric("train_precision", train_metrics["precision"], step=epoch)
            mlflow.log_metric("train_recall", train_metrics["recall"], step=epoch)

        # Validation Phase (only every `check_val_freq` epochs)
        if epoch % check_val_freq == 0:
            avg_val_loss, val_metrics = eval_model(
                model, val_loader, criterion, device, num_classes, writer, epoch)
            if log_with_mlflow:
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
                mlflow.log_metric("val_f1_score", val_metrics["f1_score"], step=epoch)
                mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
                mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)

            
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")

                return train_losses, val_losses, train_accuracies, val_accuracies


    writer.close()
    if log_with_mlflow:
        mlflow.end_run()

    return train_losses, val_losses, train_accuracies, val_accuracies

def eval_model(model, val_loader, criterion, device, num_classes, writer, epoch):
    """
    Evaluates the model on the validation dataset.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (callable): The loss function to use for evaluation.
        device (torch.device): The device to perform computations on (CPU or GPU).
        num_classes (int): The number of classes in the classification task.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple containing the average validation loss and a dictionary of validation metrics.
    """
    model.eval()
    total_val_loss, correct_val, total_val = 0, 0, 0
    all_val_predictions, all_val_labels = [], []

    with torch.no_grad():
        for patches, labels in val_loader:
            patches, labels = patches.to(device), labels.to(device)
            outputs = model(patches)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

            all_val_predictions.append(predicted)
            all_val_labels.append(labels)

    avg_val_loss = total_val_loss / len(val_loader)
    val_metrics = compute_metrics(
        torch.cat(all_val_predictions), torch.cat(all_val_labels), num_classes, device
    )

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

    # Log to MLflow
    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)

    print(f"Epoch {epoch}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
    print(f"Class-wise Metrics: {val_metrics}")

    return avg_val_loss, val_metrics

def save_model(model, epoch, location, name):
    """Saves the model to the specified location."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_epoch{epoch}_{timestamp}.pth"
    filepath = os.path.join(location, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")