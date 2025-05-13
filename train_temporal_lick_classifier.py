import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import time
import glob
from tqdm import tqdm
import cv2
import config  # Import the centralized configuration
from model_architecture import create_temporal_model  # Import the temporal model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
CONFIG = {
    "batch_size": config.TEMPORAL_BATCH_SIZE,
    "num_workers": config.NUM_WORKERS,
    "epochs": config.EPOCHS,
    "learning_rate": config.LEARNING_RATE,
    "weight_decay": config.WEIGHT_DECAY,
    "model_save_path": config.TEMPORAL_MODEL_DIR,
    "tensorboard_log_dir": config.TEMPORAL_TENSORBOARD_LOG_DIR,
    "animal_id": config.ANIMAL_ID,
    "view": config.VIEW,
    "threshold": config.THRESHOLD,
    "image_size": config.IMAGE_SIZE,
    "sequence_length": config.SEQUENCE_LENGTH,
    "lstm_hidden_size": config.LSTM_HIDDEN_SIZE,
    "lstm_num_layers": config.LSTM_NUM_LAYERS,
    "sequence_stride": config.SEQUENCE_STRIDE,
    "pos_weight": None,  # Will be set manually later
    "log_interval": 10,  # Log metrics to TensorBoard every N gradient updates
}

# Create directories
os.makedirs(CONFIG["model_save_path"], exist_ok=True)
os.makedirs(CONFIG["tensorboard_log_dir"], exist_ok=True)

# Custom Dataset class for loading frame sequences
class LickSequenceDataset(Dataset):
    def __init__(self, folders_labels_dict, transform=None, threshold=0.5,
                 sequence_length=10, sequence_stride=1):
        self.transform = transform
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.sequences = []  # Will store (folder_path, [frame_indices], label)
        skipped_folders = []

        for folder, labels in tqdm(folders_labels_dict.items(), desc="Building dataset"):
            frame_files = sorted(glob.glob(os.path.join(folder, "frame_*.jpg")))
            if len(frame_files) != len(labels):
                skipped_folders.append((folder, len(frame_files), len(labels)))
                continue
            if len(frame_files) < sequence_length:
                skipped_folders.append((folder, len(frame_files), "too short for sequence"))
                continue

            for i in range(0, len(frame_files) - sequence_length + 1, sequence_stride):
                frame_indices = list(range(i, i + sequence_length))
                sequence_label = labels[i + sequence_length - 1]
                binary_label = 1.0 if sequence_label >= threshold else 0.0
                self.sequences.append((folder, frame_indices, binary_label))

        if skipped_folders:
            print(f"Skipped {len(skipped_folders)} folders due to issues:")
            for folder, frame_count, reason in skipped_folders[:5]:
                print(f"  {os.path.basename(folder)}: {frame_count} frames, {reason}")
            if len(skipped_folders) > 5:
                print(f"  ... and {len(skipped_folders) - 5} more")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        folder_path, frame_indices, label = self.sequences[idx]
        sequence = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(folder_path, f"frame_{frame_idx:06d}.jpg")
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sequence.append(image)
        sequence_tensor = torch.stack(sequence)
        return sequence_tensor, torch.tensor(label, dtype=torch.float32)

# Collate function
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_tensor = torch.stack(sequences)
    labels_tensor = torch.stack([l.clone().detach() for l in labels]) # Ensure labels are stacked correctly
    return sequences_tensor, labels_tensor

# Training function
def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler,
               num_epochs, device, save_dir, log_dir):
    best_val_accuracy = 0.0
    global_step = 0
    accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS
    effective_batch_size = config.TEMPORAL_BATCH_SIZE * accumulation_steps

    print(f"Training with gradient accumulation:")
    print(f"- Actual batch size: {config.TEMPORAL_BATCH_SIZE}")
    print(f"- Accumulation steps: {accumulation_steps}")
    print(f"- Effective batch size: {effective_batch_size}")

    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        epoch_train_preds = []
        epoch_train_labels = []

        optimizer.zero_grad()
        progress_bar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, (sequences, labels) in enumerate(progress_bar_train):
            sequences = sequences.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(sequences).squeeze(-1)  # Ensure consistent dimension by only squeezing the last dim
                
                # Handle shape issues when batch size is 1
                if outputs.dim() == 0:  # If output is a scalar (0-dim tensor)
                    outputs = outputs.unsqueeze(0)  # Add batch dimension back
                if labels.dim() == 0:   # If labels is a scalar (0-dim tensor)
                    labels = labels.unsqueeze(0)  # Add batch dimension back
                    
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            # Store outputs and labels from *before* gradient step for metric calculation later
            if 'accumulated_outputs' not in locals():
                accumulated_outputs = []
                accumulated_labels = []
            accumulated_outputs.append(outputs.detach())
            accumulated_labels.append(labels.detach())


            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # --- Calculate Metrics for Accumulated Batches ---
                with torch.no_grad():
                    # Concatenate results from accumulated steps
                    batch_outputs = torch.cat(accumulated_outputs)
                    batch_labels = torch.cat(accumulated_labels)
                    
                    # Handle shape issues for last batch
                    if batch_outputs.dim() == 0:
                        batch_outputs = batch_outputs.unsqueeze(0)
                    if batch_labels.dim() == 0:
                        batch_labels = batch_labels.unsqueeze(0)
                        
                    preds = (batch_outputs > 0.5).float()

                    # Append to epoch lists for integrated metrics
                    epoch_train_labels.extend(batch_labels.cpu().numpy())
                    epoch_train_preds.extend(preds.cpu().numpy())

                    # Calculate integrated epoch metrics so far
                    current_epoch_labels_np = np.array(epoch_train_labels)
                    current_epoch_preds_np = np.array(epoch_train_preds)

                    current_epoch_acc = 100. * accuracy_score(current_epoch_labels_np, current_epoch_preds_np)
                    current_epoch_pos_mask = (current_epoch_labels_np == 1.0)
                    current_epoch_neg_mask = (current_epoch_labels_np == 0.0)
                    current_epoch_pos_acc = 100. * accuracy_score(current_epoch_labels_np[current_epoch_pos_mask], current_epoch_preds_np[current_epoch_pos_mask]) if current_epoch_pos_mask.sum() > 0 else 0
                    current_epoch_neg_acc = 100. * accuracy_score(current_epoch_labels_np[current_epoch_neg_mask], current_epoch_preds_np[current_epoch_neg_mask]) if current_epoch_neg_mask.sum() > 0 else 0

                # Update loss tracking
                train_loss += loss.item() * accumulation_steps # Use loss before division

                # Update progress bar
                progress_bar_train.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}", # Show loss for last micro-batch in step
                    'epoch_acc': f"{current_epoch_acc:.2f}%",
                    'epoch_pos': f"{current_epoch_pos_acc:.2f}%",
                    'epoch_neg': f"{current_epoch_neg_acc:.2f}%"
                })

                # Log integrated metrics to TensorBoard
                global_step += 1
                if global_step % CONFIG["log_interval"] == 0:
                    writer.add_scalar('training/iter_loss', loss.item() * accumulation_steps, global_step)
                    writer.add_scalar('training/epoch_integrated_accuracy', current_epoch_acc, global_step)
                    writer.add_scalar('training/epoch_integrated_pos_accuracy', current_epoch_pos_acc, global_step)
                    writer.add_scalar('training/epoch_integrated_neg_accuracy', current_epoch_neg_acc, global_step)

                # Reset accumulators for next step
                accumulated_outputs = []
                accumulated_labels = []


            # Clear cache periodically
            if (batch_idx + 1) % 50 == 0:
                 del sequences, labels, outputs, loss
                 torch.cuda.empty_cache()


        # --- Calculate Final Training Epoch Metrics ---
        final_train_labels_np = np.array(epoch_train_labels)
        final_train_preds_np = np.array(epoch_train_preds)
        epoch_loss_avg = train_loss / global_step # Average loss over steps in epoch

        final_epoch_acc = 100. * accuracy_score(final_train_labels_np, final_train_preds_np)
        final_epoch_pos_mask = (final_train_labels_np == 1.0)
        final_epoch_neg_mask = (final_train_labels_np == 0.0)
        final_epoch_pos_acc = 100. * accuracy_score(final_train_labels_np[final_epoch_pos_mask], final_train_preds_np[final_epoch_pos_mask]) if final_epoch_pos_mask.sum() > 0 else 0
        final_epoch_neg_acc = 100. * accuracy_score(final_train_labels_np[final_epoch_neg_mask], final_train_preds_np[final_epoch_neg_mask]) if final_epoch_neg_mask.sum() > 0 else 0

        print(f"Train Loss: {epoch_loss_avg:.4f} | Acc: {final_epoch_acc:.2f}% | Pos Acc: {final_epoch_pos_acc:.2f}% | Neg Acc: {final_epoch_neg_acc:.2f}%")
        writer.add_scalar('training/epoch_final_loss', epoch_loss_avg, epoch)
        writer.add_scalar('training/epoch_final_accuracy', final_epoch_acc, epoch)
        writer.add_scalar('training/epoch_final_pos_accuracy', final_epoch_pos_acc, epoch)
        writer.add_scalar('training/epoch_final_neg_accuracy', final_epoch_neg_acc, epoch)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        epoch_val_preds = []
        epoch_val_labels = []

        with torch.no_grad():
            progress_bar_val = tqdm(val_dataloader, desc="Validation")
            for batch_idx, (sequences, labels) in enumerate(progress_bar_val):
                sequences = sequences.to(device)
                labels = labels.to(device)

                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(sequences).squeeze(-1)  # Ensure consistent dimension by only squeezing the last dim
                    
                    # Handle shape issues when batch size is 1
                    if outputs.dim() == 0:  # If output is a scalar (0-dim tensor)
                        outputs = outputs.unsqueeze(0)  # Add batch dimension back
                    if labels.dim() == 0:   # If labels is a scalar (0-dim tensor)
                        labels = labels.unsqueeze(0)  # Add batch dimension back
                    
                    loss = criterion(outputs, labels)
                    # No need to divide by accumulation_steps in validation
                
                preds = (outputs.detach() > 0.5).float()
                epoch_val_labels.extend(labels.cpu().numpy())
                epoch_val_preds.extend(preds.cpu().numpy())

                # Calculate integrated epoch metrics so far
                current_val_labels_np = np.array(epoch_val_labels)
                current_val_preds_np = np.array(epoch_val_preds)

                current_val_epoch_acc = 100. * accuracy_score(current_val_labels_np, current_val_preds_np)
                current_val_pos_mask = (current_val_labels_np == 1.0)
                current_val_neg_mask = (current_val_labels_np == 0.0)
                current_val_pos_acc = 100. * accuracy_score(current_val_labels_np[current_val_pos_mask], current_val_preds_np[current_val_pos_mask]) if current_val_pos_mask.sum() > 0 else 0
                current_val_neg_acc = 100. * accuracy_score(current_val_labels_np[current_val_neg_mask], current_val_preds_np[current_val_neg_mask]) if current_val_neg_mask.sum() > 0 else 0

                val_loss += loss.item()

                # Update progress bar
                progress_bar_val.set_postfix({
                     'avg_loss': f"{val_loss / (batch_idx + 1):.4f}",
                     'epoch_acc': f"{current_val_epoch_acc:.2f}%",
                     'epoch_pos': f"{current_val_pos_acc:.2f}%",
                     'epoch_neg': f"{current_val_neg_acc:.2f}%"
                 })

                # Log integrated validation metrics less frequently
                val_step = epoch * len(val_dataloader) + batch_idx # Use a consistent step counter
                if batch_idx % CONFIG["log_interval"] == 0:
                     writer.add_scalar('validation/epoch_integrated_accuracy', current_val_epoch_acc, val_step)
                     writer.add_scalar('validation/epoch_integrated_pos_accuracy', current_val_pos_acc, val_step)
                     writer.add_scalar('validation/epoch_integrated_neg_accuracy', current_val_neg_acc, val_step)

                # Clear cache periodically
                if (batch_idx + 1) % 50 == 0:
                     del sequences, labels, outputs, loss, preds
                     torch.cuda.empty_cache()

        # --- Calculate Final Validation Epoch Metrics ---
        final_val_labels_np = np.array(epoch_val_labels)
        final_val_preds_np = np.array(epoch_val_preds)
        final_val_loss = val_loss / len(val_dataloader)

        final_val_acc = 100. * accuracy_score(final_val_labels_np, final_val_preds_np)
        final_val_pos_mask = (final_val_labels_np == 1.0)
        final_val_neg_mask = (final_val_labels_np == 0.0)
        final_val_pos_acc = 100. * accuracy_score(final_val_labels_np[final_val_pos_mask], final_val_preds_np[final_val_pos_mask]) if final_val_pos_mask.sum() > 0 else 0
        final_val_neg_acc = 100. * accuracy_score(final_val_labels_np[final_val_neg_mask], final_val_preds_np[final_val_neg_mask]) if final_val_neg_mask.sum() > 0 else 0

        print(f"Val Loss: {final_val_loss:.4f} | Acc: {final_val_acc:.2f}% | Pos Acc: {final_val_pos_acc:.2f}% | Neg Acc: {final_val_neg_acc:.2f}%")
        writer.add_scalar('validation/epoch_final_loss', final_val_loss, epoch)
        writer.add_scalar('validation/epoch_final_accuracy', final_val_acc, epoch)
        writer.add_scalar('validation/epoch_final_pos_accuracy', final_val_pos_acc, epoch)
        writer.add_scalar('validation/epoch_final_neg_accuracy', final_val_neg_acc, epoch)

        # Scheduler step (based on validation accuracy)
        scheduler.step(final_val_acc)

        # --- Save Checkpoints ---
        # Save after every epoch
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)  # Ensure directory exists
        # Save in checkpoints directory
        checkpoint_path = os.path.join(save_dir, f'checkpoints/temporal_epoch_{epoch+1}.pth')
        # Also save in top-level directory with epoch number
        top_level_path = os.path.join(save_dir, f'temporal_lick_classifier_epoch_{epoch+1}.pth')
        
        model_save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'loss': final_val_loss,
            'accuracy': final_val_acc,
            'pos_accuracy': final_val_pos_acc,
            'neg_accuracy': final_val_neg_acc,
        }
        
        # Save to both locations
        torch.save(model_save_dict, checkpoint_path)
        torch.save(model_save_dict, top_level_path)
        print(f"Model saved to {top_level_path} and {checkpoint_path}")

        # Save best model based on validation accuracy
        if final_val_acc > best_val_accuracy:
            best_val_accuracy = final_val_acc
            best_model_path = os.path.join(save_dir, 'temporal_lick_classifier_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': final_val_loss,
                'accuracy': final_val_acc,
            }, best_model_path)
            print(f"Saved best model to {best_model_path} (Val Acc: {final_val_acc:.2f}%)")

        # Clean up CUDA memory at end of epoch
        torch.cuda.empty_cache()

    writer.close()
    print(f"Training finished. Best validation accuracy: {best_val_accuracy:.2f}%")
    return model

# Function to calculate positive class weight for imbalanced data
def calculate_pos_weight(dataset):
    """Calculate positive class weight inversely proportional to class frequencies"""
    labels = np.array([sample[2] for sample in dataset.sequences])
    neg_count = np.sum(labels == 0)
    pos_count = np.sum(labels == 1)
    
    # Avoid division by zero
    if pos_count == 0:
        return torch.tensor([1.0]).to(device)
    
    # Calculate weight as ratio of negative to positive samples
    return torch.tensor([neg_count / pos_count]).to(device)

# Main function
def main():
    train_labels_path = config.get_train_labels_path()
    test_labels_path = config.get_test_labels_path()
    print(f"Loading data from: {train_labels_path} and {test_labels_path}")
    with open(train_labels_path, 'rb') as f: train_labels_dict = pickle.load(f)
    with open(test_labels_path, 'rb') as f: test_labels_dict = pickle.load(f)
    print(f"Loaded {len(train_labels_dict)} training folders and {len(test_labels_dict)} test folders")

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Added ColorJitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = LickSequenceDataset(
        train_labels_dict, transform=train_transform, threshold=CONFIG["threshold"],
        sequence_length=CONFIG["sequence_length"], sequence_stride=CONFIG["sequence_stride"]
    )
    val_dataset = LickSequenceDataset(
        test_labels_dict, transform=val_transform, threshold=CONFIG["threshold"],
        sequence_length=CONFIG["sequence_length"], sequence_stride=CONFIG["sequence_stride"]
    )
    print(f"Training dataset: {len(train_dataset)} sequences")
    print(f"Validation dataset: {len(val_dataset)} sequences")

    # Analyze class distribution
    train_labels = [sample[2] for sample in train_dataset.sequences]
    train_labels_array = np.array(train_labels)
    train_pos = np.sum(train_labels_array == 1.0)
    train_neg = len(train_labels) - train_pos
    print(f"Training class distribution: {train_pos} positive, {train_neg} negative, ratio: {train_pos/len(train_labels):.2%}")

    # Calculate positive class weight
    CONFIG["pos_weight"] = calculate_pos_weight(train_dataset)
    print(f"Calculated positive class weight: {CONFIG['pos_weight'].item():.4f}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True, collate_fn=collate_fn, persistent_workers=True if CONFIG["num_workers"] > 0 else False # Added persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True, collate_fn=collate_fn, persistent_workers=True if CONFIG["num_workers"] > 0 else False # Added persistent_workers
    )

    # Create model
    model = create_temporal_model(
        sequence_length=CONFIG["sequence_length"], hidden_size=CONFIG["lstm_hidden_size"],
        num_layers=CONFIG["lstm_num_layers"]
    )
    model = model.to(device)
    print(f"Model architecture:")
    print(f"- Sequence length: {CONFIG['sequence_length']}")
    print(f"- LSTM hidden size: {CONFIG['lstm_hidden_size']}")
    print(f"- LSTM layers: {CONFIG['lstm_num_layers']}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=CONFIG["pos_weight"])
    optimizer = optim.AdamW( # Switched to AdamW
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Train model
    train_model(
        train_loader, val_loader, model, criterion, optimizer, scheduler,
        CONFIG["epochs"], device, CONFIG["model_save_path"], CONFIG["tensorboard_log_dir"]
    )

if __name__ == "__main__":
    # Enable anomaly detection for debugging NaN issues if they occur
    # torch.autograd.set_detect_anomaly(True)
    main() 