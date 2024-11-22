import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime

from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj, dropout_node


from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix

import mlflow
from mlflow.utils.logging_utils import disable_logging,enable_logging

#TODO switch to logging after initial tests
#TODO move earlystopping and stats calc to utils instead of haveing two copies in training 
#TODO Convert these into hyperparameters that we can add to the dataloader instead of doing as part of the training loop
#TODO Update Docstrings
#            
class EarlyStopping:
    """EarlyStopping is a utility class to stop training when a monitored metric has stopped improving.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): If True, prints a message for each validation loss improvement.
        path (str): Path to save the model when the validation loss improves.
        counter (int): Counter for the number of epochs since the last improvement.
        best_loss (float): The best validation loss observed.
        early_stop (bool): Flag indicating whether to stop training.
    simplified version of https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py
    """
    def __init__(self, patience=25, delta=0.01, verbose=True,path='best_model.pth'):
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


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.k

def train_model(model, train_loader, val_loader, criterion, optimizer, device, location,
                epochs=50, patience=25, delta=1e-8, check_val_freq=5, num_classes=2,
                model_name='model', log_with_mlflow=True, mlflow_uri="http://127.0.0.1:5000"):
    """
    Train the GNN model.

    Args:
        model (nn.Module): The GNN model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (callable): Loss function.
        optimizer (Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to perform computations on.
        location (str): Directory to save the best model.
        epochs (int): Number of training epochs.
        patience (int): Patience for early stopping.
        delta (float): Minimum change to qualify as improvement.
        check_val_freq (int): Frequency of validation checks.
        num_classes (int): Number of classes.
        model_name (str): Name of the model.
        log_with_mlflow (bool): Whether to log training with MLflow.
        mlflow_uri (str): URI for the MLflow tracking server.

    Returns:
        None
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, delta=delta,
                                   path=os.path.join(location, model_name + '_best_model.pth'))
    writer = SummaryWriter(log_dir=location)

    if log_with_mlflow:
        mlflow.set_tracking_uri(mlflow_uri)
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"Started MLflow run with ID: {run_id}")
            disable_logging()  # Disable logging inside the MLflow context

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_train_loss = 0
        all_train_predictions, all_train_labels = [], []

        for batch in train_loader:
            
            batch = batch.to(device)
            
            # EDGE Dropping
            # Drop edges with probability 0.2(hard coded) for the current batch
            batch.edge_index, _ = dropout_adj(batch.edge_index, p=0.2, training=True)

            # Node Dropping
            # Drop nodes with probability 0.2(hard coded) for the current batch
            if hasattr(batch, "x"):
                batch.x, batch.edge_index, _ = dropout_node(batch.x, batch.edge_index, p=0.2, training=True)   
            
            
            outputs = model(batch.x, batch.edge_index, batch.batch)
            optimizer.zero_grad()
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_train_predictions.append(predicted)
            all_train_labels.append(batch.y)

        avg_train_loss = total_train_loss / len(train_loader)
        train_metrics = compute_metrics(torch.cat(all_train_predictions), torch.cat(all_train_labels), num_classes, device)

        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f},  Acc: {train_metrics['accuracy']:.2f}%,  F1: {train_metrics['f1_score']},  Precision: {train_metrics['precision']:.2f},  Recall: {train_metrics['recall']:.2f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("F1/train", train_metrics["f1_score"], epoch)

        if log_with_mlflow:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_metrics["accuracy"], step=epoch)
                mlflow.log_metric("train_f1_score", train_metrics["f1_score"], step=epoch)
                mlflow.log_metric("train_precision", train_metrics["precision"], step=epoch)
                mlflow.log_metric("train_recall", train_metrics["recall"], step=epoch)

        # Validation phase
        if epoch % check_val_freq == 0:
            avg_val_loss, val_metrics = eval_model(model, val_loader, criterion, device, num_classes, epoch)
            if log_with_mlflow:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
                    mlflow.log_metric("val_f1_score", val_metrics["f1_score"], step=epoch)
                    mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
                    mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    writer.close()
    if log_with_mlflow:
        mlflow.end_run()


def eval_model(model, data_loader, criterion, device, num_classes, epoch):
    """
    Evaluates the GNN model on the validation dataset.

    This function evaluates the model on the validation dataset using the provided data loader and loss function.
    It also computes the validation metrics and prints the validation loss and accuracy.

    Args:
        model (nn.Module): The GNN model to evaluate.
        data_loader (DataLoader): DataLoader for the validation dataset.
        criterion (callable): Loss function.
        device (torch.device): Device to perform computations on.
        num_classes (int): Number of classes.
        epoch (int): Current epoch number.

    Returns:
        tuple: Average validation loss and validation metrics.
    """
    model.eval()
    total_val_loss = 0
    all_val_predictions, all_val_labels,all_val_conf = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y)

            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_val_predictions.append(predicted)
            all_val_labels.append(batch.y)

            confidences = F.softmax(outputs, dim=1).max(dim=1).values
            all_val_conf.append(confidences)


    avg_val_loss = total_val_loss / len(data_loader)
    val_metrics = compute_metrics(
        torch.cat(all_val_predictions), torch.cat(all_val_labels), num_classes, device
    )
    all_val_conf = torch.cat(all_val_conf)
    avg_confidence = all_val_conf.mean().item()


    print(f"Epoch {epoch}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, Val F1: {val_metrics['f1_score']}, Val Precision: {val_metrics['precision']:.2f}, Val Recall: {val_metrics['recall']:.2f}")
    print(f"Average Prediction Confidence: {avg_confidence:.4f}")
    print(f"Class-wise Metrics: ")
    
    return avg_val_loss, val_metrics
