# train_utils.py

import torch

import torch
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5,check_val_freq=5):
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels,_,fov_paths in train_loader:
            labels=labels.to(device)
            print(labels)
            print(fov_paths[0])
            for patches in inputs:
                patch = patches.to(device).float()
                optimizer.zero_grad()
                outputs = model(patch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Compute accuracy for training set
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)#Adjust for length of loader
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Evaluate on validation set
        if (epoch%check_val_freq==0) and epoch>0:
            val_loss, val_accuracy = eval_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")

        print(f'Epoch {epoch}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():

        for inputs, labels,_ in dataloader:
            labels=labels.to(device)
            for patches in inputs:
                patch = patches.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)#Needs to be adjusted for the number of patches
        accuracy = 100 * correct / total
        return avg_loss, accuracy

def dataset_overlap(df1,df2,col):
  #set up for polars dataframes   
    set1 = set(df1[col].to_list())
    set2 = set(df2[col].to_list())

    if set1 & set2:
        print("There is an overlap in patient numbers.")
    else:
        print("No overlap in patient numbers.")

