import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, labels, smooth=1):
        preds = torch.sigmoid(logits)  # Use torch.sigmoid to ensure compatibility

        # Flatten label and prediction tensors
        preds = preds.view(-1)
        labels = labels.view(-1)


        intersection = (preds * labels).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + labels.sum() + smooth)

        return 1 - dice  
    
def train_epoch(model, optimizer, loss_func, dataloader, device, use_distance_maps, use_dems, use_ndvi):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    # Determine indices to drop
    drop_indices = []
    if not use_distance_maps:
        drop_indices.append(5)  # Index 5 is for distance maps
    if not use_dems:
        drop_indices.append(6)  # Index 6 is for DEMs
    if not use_ndvi:
        drop_indices.append(7)  # Index 7 is for NDVI

    for data_batch, labels_batch, _, _ in dataloader:
        # Exclude specified bands based on flags
        if drop_indices:
            data_batch = data_batch[:, [i for i in range(data_batch.shape[1]) if i not in drop_indices], :, :]

        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        y_hat = model(data_batch)
        loss = loss_func(y_hat, labels_batch)
        iou = calculate_iou(y_hat, labels_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data_batch.size(0)
        running_iou += iou.item() * data_batch.size(0)

    train_epoch_loss = running_loss / len(dataloader.dataset)
    train_epoch_iou = running_iou / len(dataloader.dataset)
    print(f'Train - Loss: {train_epoch_loss:.4f}, IoU: {train_epoch_iou:.4f}')

    return train_epoch_loss, train_epoch_iou

def val_epoch(model, loss_func, dataloader, device, use_distance_maps, use_dems, use_ndvi):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_iou = 0.0

    # Initialize lists to store true labels, predictions, and tile IDs
    all_y_true, all_y_pred, all_tile_ids = [], [], []
    
    # Determine indices to drop
    drop_indices = []
    if not use_distance_maps:
        drop_indices.append(5)  # Index 5 is for distance maps
    if not use_dems:
        drop_indices.append(6)  # Index 6 is for DEMs
    if not use_ndvi:
        drop_indices.append(7)  # Index 7 is for NDVI

    with torch.no_grad():
        for data_batch, labels_batch, _, tile_ids_batch in dataloader:
            # Exclude specified bands based on flags
            if drop_indices:
                data_batch = data_batch[:, [i for i in range(data_batch.shape[1]) if i not in drop_indices], :, :]
            
            data_batch = data_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Forward pass
            y_hat = model(data_batch)
            loss = loss_func(y_hat, labels_batch)
            iou = calculate_iou(y_hat, labels_batch)

            # Aggregate loss and IoU
            running_loss += loss.item() * data_batch.size(0)
            running_iou += iou.item() * data_batch.size(0)

            # Store true labels and predictions
            all_y_true.append(labels_batch.cpu().numpy())
            all_y_pred.append(y_hat.cpu().numpy())
            all_tile_ids.extend(tile_ids_batch)

    # Calculate average loss and IoU for the epoch
    val_epoch_loss = running_loss / len(dataloader.dataset)
    val_epoch_iou = running_iou / len(dataloader.dataset)
    print(f'Eval - Loss: {val_epoch_loss:.4f}, IoU: {val_epoch_iou:.4f}')

    # Concatenate arrays of true labels and predictions
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    return val_epoch_loss, val_epoch_iou, all_y_true, all_y_pred, all_tile_ids

def load_checkpoint(outputs, model, optimizer, scheduler):
    
    # Set up checkpoints path
    checkpoints_path =  os.path.join(outputs, 'checkpoints')
    
    filepath = os.path.join(checkpoints_path, f'{model.modelname}.pth')
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load the scheduler state only if scheduler is not None and the state is saved in the checkpoint
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
        min_val_loss = checkpoint['min_val_loss']
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']
        train_iou_history = checkpoint['train_iou_history']
        val_iou_history = checkpoint['val_iou_history']
        best_epoch = checkpoint.get('best_epoch', checkpoint['epoch'])  # Load best_epoch, default to current epoch if not found
        print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
        return start_epoch, min_val_loss, train_loss_history, val_loss_history, train_iou_history, val_iou_history, best_epoch
    else:
        print(f"No checkpoint found at '{filepath}', starting from scratch")
        return 1, np.inf, [], [], [], [], 0  # Include default best_epoch

def train_model(model, optimizer, loss_func, scheduler, train_dataloader, val_dataloader, device, num_epochs, outputs, use_distance_maps, use_dems, use_ndvi, use_checkpoint=False):
    since = time.time()
    model.to(device)
    
    # Set up checkpoints path
    checkpoints_path =  os.path.join(outputs, 'checkpoints')

    # Initialize training variables
    start_epoch, min_val_loss, train_loss_history, val_loss_history, train_iou_history, val_iou_history = 1, np.inf, [], [], [], []
    counter = 0
    patience = 10
    delta_p = 0.001
    best_epoch = 0
    best_checkpoint_path = os.path.join(checkpoints_path, f'{model.modelname}.pth')

    # Load checkpoint if specified and exists
    if use_checkpoint:
        start_epoch, min_val_loss, train_loss_history, val_loss_history, train_iou_history, val_iou_history, best_epoch = load_checkpoint(checkpoints_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # Training phase
        train_epoch_loss, train_epoch_iou = train_epoch(model, optimizer, loss_func, train_dataloader, device, use_distance_maps, use_dems, use_ndvi)
        train_loss_history.append(train_epoch_loss)
        train_iou_history.append(train_epoch_iou)

        # Validation phase
        val_epoch_loss, val_epoch_iou, _, _, _ = val_epoch(model, loss_func, val_dataloader, device, use_distance_maps, use_dems, use_ndvi)
        val_loss_history.append(val_epoch_loss)
        val_iou_history.append(val_epoch_iou)

        # Update the learning rate according to the scheduler, if it exists
        if scheduler is not None:
            scheduler.step()

        # Checkpoint
        if val_epoch_loss < min_val_loss - delta_p:
            min_val_loss = val_epoch_loss
            best_epoch = epoch
            counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,  # Save the scheduler state only if it exists
                'min_val_loss': min_val_loss,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'train_iou_history': train_iou_history,
                'val_iou_history': val_iou_history,
                'best_epoch': best_epoch
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Checkpoint saved at: {best_checkpoint_path}")
        else:
            counter += 1

        if counter == patience:
            print(f'\nEarly stopping after {patience} epochs without improvement.')
            break

    # Training Time Calculation
    time_elapsed = time.time() - since
    time_epoch = time_elapsed / num_epochs
    time_epoch = f'{time_epoch // 60:.0f}m {time_epoch % 60:.0f}s'
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Load best model weights before returning
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path)['model_state_dict'])

    return model, time_epoch, train_loss_history, val_loss_history, train_iou_history, val_iou_history, best_epoch

# IOU
def calculate_iou(logits, labels, p_threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(logits) > p_threshold).float()

    preds = preds.view(-1)
    labels = labels.view(-1)

    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection

    IoU = (intersection + smooth) / (union + smooth)
    
    return IoU


def save_learning_curves(train_loss, val_loss, train_iou, val_iou, best_epoch, outputs):    
    # Create subplots with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Adjust epoch values for the training data to shift the curves to the left by 0.5
    epochs_train = [x - 0.5 for x in range(1, len(train_loss) + 1)]
    epochs_val = range(1, len(val_loss) + 1)  # Validation epochs remain unchanged

    # Plot Training and Validation Loss on the first axis
    ax1.set_title('Training and Validation Loss')
    ax1.plot(epochs_train, train_loss, label="Train Loss")  # Use adjusted epochs for training data
    ax1.plot(epochs_val, val_loss, label="Validation Loss")
    ax1.axvline(x=best_epoch, color='grey', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Training and Validation IoU on the second axis
    ax2.set_title('Training and Validation IoU')
    ax2.plot(epochs_train, train_iou, label="Train IoU")  # Use adjusted epochs for training data
    ax2.plot(epochs_val, val_iou, label="Validation IoU")
    ax2.axvline(x=best_epoch, color='grey', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('IoU')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure with a specified filename in the given directory path
    output_path = os.path.join(outputs, 'learning_curves.png')
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")