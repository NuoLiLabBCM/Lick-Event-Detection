#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create labeled videos showing model predictions vs ground truth.
This script loads frames from a folder, runs the model on them,
and creates a video with the predictions and ground truth visualized.
"""

import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import random

# Set random seeds based on current time for different results each run
current_time = int(time.time())
torch.manual_seed(current_time)
np.random.seed(current_time)
random.seed(current_time)

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using random seed: {current_time} (changes each run)")

# Configuration
ANIMAL_ID = "DL005"
VIEW = "side"
MODEL_PATH = "models/temporal_lick_classifier/temporal_lick_classifier_best.pth"
OUTPUT_DIR = "labeled_videos"
FPS = 30  # Video frames per second
THRESHOLD = 0.5  # Threshold for binary classification

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_model():
    """Create the same model architecture used during training"""
    model = models.resnet18(weights=None)  # No pretrained weights for inference
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

def load_trained_model(model_path):
    """Load a trained model checkpoint"""
    try:
        # Try to import and use temporal model architecture first
        from model_architecture import create_temporal_model
        
        # Create temporal model with config parameters
        import config
        model = create_temporal_model(
            sequence_length=config.SEQUENCE_LENGTH,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    except (ImportError, KeyError) as e:
        print(f"Falling back to basic model: {e}")
        # Fallback to old architecture if needed
        model = create_model()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Get epoch from checkpoint if available
    epoch = checkpoint.get('epoch', 'unknown')
    accuracy = checkpoint.get('val_accuracy', checkpoint.get('accuracy', 0.0))
    print(f"Loaded model from epoch {epoch} with validation accuracy: {accuracy:.4f}")
    
    return model

def get_frame_predictions(model, frames_folder, batch_size=8):
    """Run inference on all frames in a folder"""
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frames_folder, "frame_*.jpg")))
    
    if not frame_files:
        print(f"No frames found in {frames_folder}")
        return [], []
    
    # Check if we're using a temporal model by looking at the model's class name
    is_temporal = 'Temporal' in model.__class__.__name__
    
    if is_temporal:
        print("Using temporal model for prediction")
        return get_temporal_predictions(model, frame_files, batch_size)
    else:
        print("Using frame-based model for prediction")
        return get_basic_predictions(model, frame_files, batch_size)

def get_basic_predictions(model, frame_files, batch_size=32):
    """Run inference on individual frames for a basic model"""
    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process frames in batches
    predictions = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(frame_files), batch_size), desc="Processing frames"):
        batch_files = frame_files[i:i+batch_size]
        batch_images = []
        
        # Preprocess each image
        for frame_file in batch_files:
            image = Image.open(frame_file).convert('RGB')
            image = transform(image)
            batch_images.append(image)
        
        # Convert to tensor and move to device
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Run inference
        with torch.no_grad():
            batch_preds = model(batch_tensor).squeeze().cpu().numpy()
            
            # Handle the case where there's only one prediction
            if not isinstance(batch_preds, np.ndarray):
                batch_preds = np.array([batch_preds])
                
            predictions.extend(batch_preds)
    
    return frame_files, predictions

def get_temporal_predictions(model, frame_files, batch_size=8):
    """Run inference using a temporal model on sequences of frames"""
    # Get sequence parameters from config
    try:
        import config
        sequence_length = config.SEQUENCE_LENGTH
        sequence_stride = 1  # Always use stride 1 for prediction
    except ImportError:
        sequence_length = 30  # Default if config not available
        sequence_stride = 1
    
    print(f"Temporal model settings: sequence_length={sequence_length}, stride={sequence_stride}")
    
    # Create sequences from frames
    sequences = []
    sequence_indices = []  # Store the frame index each sequence prediction belongs to
    
    # Skip if not enough frames
    if len(frame_files) < sequence_length:
        print(f"Warning: Only {len(frame_files)} frames available, which is less than the required sequence length {sequence_length}")
        return frame_files, [0.0] * len(frame_files)  # Return zeros for all frames
    
    # Handle frames at the beginning (less than sequence_length-1 frames before them)
    # For frames 0 to sequence_length-2, we need to pad the beginning
    padding_needed = sequence_length - 1
    
    # Create padded sequences for the first frames
    for i in range(padding_needed):
        # Determine how many frames to repeat at the beginning
        repeat_count = sequence_length - 1 - i
        
        # Create a sequence with repeated initial frames
        repeated_frames = [frame_files[0]] * repeat_count
        padded_sequence = repeated_frames + frame_files[:i+1]
        
        # Add the padded sequence and its target frame index
        sequences.append(padded_sequence)
        sequence_indices.append(i)  # This sequence prediction is for frame i
    
    # Create regular sequences for the rest of the frames
    for i in range(0, len(frame_files) - sequence_length + 1, sequence_stride):
        sequence = frame_files[i:i+sequence_length]
        sequences.append(sequence)
        sequence_indices.append(i + sequence_length - 1)  # This is the frame index this sequence prediction belongs to
    
    print(f"Created {len(sequences)} sequences")
    
    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process sequences in batches
    sequence_predictions = []
    
    # Create data loader
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing sequences"):
        batch_sequences = sequences[i:i+batch_size]
        batch_tensors = []
        
        # Preprocess each sequence
        for sequence in batch_sequences:
            # Load and transform each frame in the sequence
            frame_tensors = []
            for frame_file in sequence:
                image = Image.open(frame_file).convert('RGB')
                tensor = transform(image)
                frame_tensors.append(tensor)
            
            # Stack frames into a sequence tensor [sequence_length, channels, height, width]
            sequence_tensor = torch.stack(frame_tensors)
            batch_tensors.append(sequence_tensor)
        
        # Stack all sequences into a batch tensor [batch_size, sequence_length, channels, height, width]
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Run inference
        with torch.no_grad():
            batch_preds = model(batch_tensor).squeeze().cpu().numpy()
            
            # Apply sigmoid if needed (for raw logits)
            if np.any(batch_preds > 1.0) or np.any(batch_preds < 0.0):
                batch_preds = 1.0 / (1.0 + np.exp(-batch_preds))
            
            # Handle the case where there's only one prediction
            if not isinstance(batch_preds, np.ndarray):
                batch_preds = np.array([batch_preds])
                
            sequence_predictions.extend(batch_preds)
    
    # Initialize frame predictions with zeros
    frame_predictions = np.zeros(len(frame_files))
    
    # Directly map sequence predictions to their corresponding frames
    for seq_idx, frame_idx in enumerate(sequence_indices):
        frame_predictions[frame_idx] = sequence_predictions[seq_idx]
    
    # Apply temporal smoothing using a moving average window for smoother transitions
    window_size = 5
    frame_smoothed = np.copy(frame_predictions)
    
    for i in range(len(frame_predictions)):
        # Calculate window boundaries
        window_start = max(0, i - window_size // 2)
        window_end = min(len(frame_predictions), i + window_size // 2 + 1)
        
        # Calculate weighted moving average
        weights = np.exp(-0.5 * np.square(np.arange(window_start - i, window_end - i) / (window_size / 4)))
        frame_smoothed[i] = np.sum(frame_predictions[window_start:window_end] * weights) / np.sum(weights)
    
    print(f"Mapped {len(sequence_predictions)} sequence predictions to {len(frame_files)} frames")
    
    return frame_files, frame_smoothed

def load_ground_truth_labels(animal_id, view, frames_folder):
    """Load ground truth labels for a specific folder"""
    # Try to find the folder directly in the dictionary
    test_labels_path = f"C:/Users/Nuo Lab/Desktop/lick/withframenew/datatags_{view}_corrected/test_labels_lick_{animal_id}.pkl"
    train_labels_path = f"C:/Users/Nuo Lab/Desktop/lick/withframenew/datatags_{view}_corrected/train_labels_lick_{animal_id}.pkl"
    
    labels = None
    
    # Extract folder components
    folder_parts = os.path.normpath(frames_folder).split(os.sep)
    session_date = None
    trial_num = None
    
    # Find session date and trial number in the path
    for i, part in enumerate(folder_parts):
        if part.startswith("trial_"):
            trial_num = part
            if i > 0:  # Check if there's a part before this that could be the date
                potential_date = folder_parts[i-1]
                if "_" in potential_date:  # Simple check that it looks like a date format
                    session_date = potential_date
    
    print(f"Looking for exact match - Date: {session_date}, Trial: {trial_num}")
    
    # Try to load test labels
    try:
        with open(test_labels_path, 'rb') as f:
            test_labels = pickle.load(f)
            # Look for an exact key match first
            if frames_folder in test_labels:
                labels = test_labels[frames_folder]
                print(f"Found exact match in test dataset: {len(labels)} labels")
            else:
                # If not found, look for a key with matching date AND trial
                for key in test_labels.keys():
                    key_parts = os.path.normpath(key).split(os.sep)
                    key_has_date = False
                    key_has_trial = False
                    key_trial = None
                    
                    # Extract trial part from the key
                    for part in key_parts:
                        if part.startswith("trial_"):
                            key_trial = part
                            break
                    
                    # Check if the key contains both the right session date and EXACT trial number
                    for part in key_parts:
                        if session_date and session_date == part:
                            key_has_date = True
                        if trial_num and trial_num == key_trial:
                            key_has_trial = True
                    
                    # Only use this key if BOTH date and trial match EXACTLY
                    if key_has_date and key_has_trial:
                        labels = test_labels[key]
                        print(f"Found matching date and trial in test dataset: {key}")
                        print(f"Label count: {len(labels)}")
                        break
    except Exception as e:
        print(f"Error loading test labels: {e}")
    
    # If not found in test, try training labels with the same exact matching criteria
    if labels is None:
        try:
            with open(train_labels_path, 'rb') as f:
                train_labels = pickle.load(f)
                # Look for an exact key match first
                if frames_folder in train_labels:
                    labels = train_labels[frames_folder]
                    print(f"Found exact match in training dataset: {len(labels)} labels")
                else:
                    # If not found, look for a key with matching date AND trial
                    for key in train_labels.keys():
                        key_parts = os.path.normpath(key).split(os.sep)
                        key_has_date = False
                        key_has_trial = False
                        key_trial = None
                        
                        # Extract trial part from the key
                        for part in key_parts:
                            if part.startswith("trial_"):
                                key_trial = part
                                break
                        
                        # Check if the key contains both the right session date and EXACT trial number
                        for part in key_parts:
                            if session_date and session_date == part:
                                key_has_date = True
                            if trial_num and trial_num == key_trial:
                                key_has_trial = True
                        
                        # Only use this key if BOTH date and trial match EXACTLY
                        if key_has_date and key_has_trial:
                            labels = train_labels[key]
                            print(f"Found matching date and trial in training dataset: {key}")
                            print(f"Label count: {len(labels)}")
                            break
        except Exception as e:
            print(f"Error loading training labels: {e}")
    
    if labels is None:
        print(f"Could not find labels for folder: {frames_folder}")
    else:
        print(f"Using labels for path: {frames_folder}")
    
    return labels

def find_frame_folders(animal_id, max_folders=500):
    """Find folders containing frame_*.jpg files"""
    base_dir = f"I:/videos/frames/side/{animal_id}"
    
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        print("Checking alternate paths...")
        
        # Try common alternate paths
        alternate_paths = [
            f"C:/Users/Nuo Lab/Desktop/lick_proj/G/videos/frames/{animal_id}",
            f"C:/Users/Nuo Lab/Desktop/{animal_id}",
            f"C:/Users/Nuo Lab/Desktop/lick_proj/{animal_id}",
            f"I:/frames/side/{animal_id}"
        ]
        
        for path in alternate_paths:
            if os.path.exists(path):
                print(f"Found alternate path: {path}")
                base_dir = path
                break
    
    if not os.path.exists(base_dir):
        print(f"Could not find a valid base directory for animal {animal_id}")
        return []
    
    # Find all trial directories
    trial_folders = []
    
    # First check if there are any session date directories
    session_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    for session_dir in session_dirs:
        # Look for trial_X folders in each session directory
        trial_dirs = [d for d in glob.glob(os.path.join(session_dir, "trial_*")) if os.path.isdir(d)]
        for trial_dir in trial_dirs:
            # Check if the folder contains frame_*.jpg files
            frame_files = glob.glob(os.path.join(trial_dir, "frame_*.jpg"))
            if frame_files:
                trial_folders.append(trial_dir)
                if len(trial_folders) >= max_folders:
                    break
        
        if len(trial_folders) >= max_folders:
            break
    
    # If we haven't found any, try using the datatags paths directly
    if not trial_folders:
        # Try to load train and test label dictionaries to get valid paths
        view = "side"  # Default view
        test_labels_path = f"C:/Users/Nuo Lab/Desktop/lick/withframenew/datatags_{view}_corrected/test_labels_lick_{animal_id}.pkl"
        train_labels_path = f"C:/Users/Nuo Lab/Desktop/lick/withframenew/datatags_{view}_corrected/train_labels_lick_{animal_id}.pkl"
        
        try:
            with open(test_labels_path, 'rb') as f:
                test_labels = pickle.load(f)
                for folder in test_labels.keys():
                    if os.path.exists(folder) and glob.glob(os.path.join(folder, "frame_*.jpg")):
                        trial_folders.append(folder)
                        if len(trial_folders) >= max_folders:
                            break
        except Exception as e:
            print(f"Error loading test labels: {e}")
            
        if len(trial_folders) < max_folders:
            try:
                with open(train_labels_path, 'rb') as f:
                    train_labels = pickle.load(f)
                    for folder in train_labels.keys():
                        if os.path.exists(folder) and glob.glob(os.path.join(folder, "frame_*.jpg")):
                            trial_folders.append(folder)
                            if len(trial_folders) >= max_folders:
                                break
            except Exception as e:
                print(f"Error loading training labels: {e}")
    
    return trial_folders

def create_labeled_video(frames_folder, frame_files, predictions, labels, output_path):
    """
    Create a video with frames and visualization of predictions and ground truth
    
    Args:
        frames_folder (str): Path to the folder containing frames
        frame_files (list): List of frame file paths
        predictions (list): Model predictions for each frame
        labels (list): Ground truth labels for each frame
        output_path (str): Path to save the output video
    """
    if not frame_files:
        print(f"No frames to process for {frames_folder}")
        return
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Determine if we have real labels or dummy labels
    has_real_labels = not np.all(labels == 0.0)
    
    # Make sure lengths match
    min_len = min(len(frame_files), len(predictions), len(labels))
    frame_files = frame_files[:min_len]
    predictions = predictions[:min_len]
    labels = labels[:min_len]
    
    # Get frame dimensions from the first frame
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Error reading first frame: {frame_files[0]}")
        return
    
    h, w, _ = first_frame.shape
    
    # Define the visualization area dimensions
    vis_height = 100  # Height of the visualization area
    total_height = h + vis_height
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (w, total_height))
    
    # Process each frame
    for i, frame_file in enumerate(tqdm(frame_files, desc="Creating video")):
        # Read the original frame
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Error reading frame: {frame_file}")
            continue
        
        # Create visualization area (white background)
        vis_area = np.ones((vis_height, w, 3), dtype=np.uint8) * 255
        
        # Get current prediction and ground truth
        pred = predictions[i]
        gt = labels[i] if has_real_labels else None
        
        # Convert to binary using threshold
        pred_binary = 1 if pred >= THRESHOLD else 0
        gt_binary = 1 if gt is not None and gt >= THRESHOLD else 0
        
        # Draw prediction bar (blue)
        pred_width = int(w * pred)
        cv2.rectangle(vis_area, (0, 10), (pred_width, 40), (255, 0, 0), -1)
        
        # Draw ground truth bar (green) only if we have real labels
        if has_real_labels:
            gt_width = int(w * gt)
            cv2.rectangle(vis_area, (0, 50), (gt_width, 80), (0, 255, 0), -1)
        else:
            # Show text instead of bar
            gt_text = "No ground truth available"
            cv2.putText(vis_area, gt_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_area, f"Pred: {pred:.2f} ({pred_binary})", (10, 30), font, 0.5, (0, 0, 0), 1)
        
        if has_real_labels:
            cv2.putText(vis_area, f"GT: {gt:.2f} ({gt_binary})", (10, 70), font, 0.5, (0, 0, 0), 1)
        
        # Add frame number
        cv2.putText(vis_area, f"Frame: {i}", (w-150, 30), font, 0.5, (0, 0, 0), 1)
        
        # Highlight if there's a mismatch between prediction and ground truth (only if real labels)
        if has_real_labels and pred_binary != gt_binary:
            cv2.putText(vis_area, "MISMATCH", (w-150, 70), font, 0.5, (0, 0, 255), 1)
            # Add a red border to the frame
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 2)
        
        # Combine frame and visualization area
        combined = np.vstack((frame, vis_area))
        
        # Write to video
        video_writer.write(combined)
    
    # Release the VideoWriter
    video_writer.release()
    print(f"Video saved to {output_path}")
    return output_path

def main():
    # Find a model file
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        model_files = glob.glob("models/temporal_lick_classifier/*.pth")
        if not model_files:
            model_files = glob.glob("*.pth")
        
        if model_files:
            print(f"Using alternative model file: {model_files[0]}")
            model_path = model_files[0]
        else:
            print("No model file found. Please specify a valid model path.")
            return
    else:
        model_path = MODEL_PATH
    
    # Load the model
    try:
        model = load_trained_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find available frame folders
    frame_folders = find_frame_folders(ANIMAL_ID)
    if not frame_folders:
        print(f"No frame folders found for animal {ANIMAL_ID}")
        return
    
    print(f"Found {len(frame_folders)} frame folders")
    
    # Process each folder to check if it has valid labels
    valid_folders = []
    for folder in frame_folders:
        labels = load_ground_truth_labels(ANIMAL_ID, VIEW, folder)
        if labels is not None:
            valid_folders.append(folder)
    
    # If we found folders with labels, use them
    if valid_folders:
        print(f"Found {len(valid_folders)} folders with labels")
        # Randomly select a folder
        selected_folder = random.choice(valid_folders)
        print(f"Selected folder: {selected_folder}")
        
        # Get the frame files and predictions
        frame_files, predictions = get_frame_predictions(model, selected_folder)
        
        # Get the ground truth labels
        labels = load_ground_truth_labels(ANIMAL_ID, VIEW, selected_folder)
        
        # Create the output video path
        folder_name = os.path.basename(selected_folder)
        session_name = os.path.basename(os.path.dirname(selected_folder))
        output_filename = f"{ANIMAL_ID}_{session_name}_{folder_name}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Create the labeled video
        video_path = create_labeled_video(selected_folder, frame_files, predictions, labels, output_path)
        
        if video_path:
            print(f"\nVideo created successfully!")
            print(f"Video saved to: {video_path}")
        else:
            print("Failed to create video")
    else:
        # If no folders have labels, just run the model on a random folder to see the predictions
        print("No folders with labels found. Running model on a random folder without ground truth.")
        # Randomly select a folder
        selected_folder = random.choice(frame_folders)
        print(f"Selected folder: {selected_folder}")
        
        # Get the frame files and predictions
        frame_files, predictions = get_frame_predictions(model, selected_folder)
        
        # Create dummy labels (all zeros)
        dummy_labels = [0.0] * len(predictions)
        
        # Create the output video path
        folder_name = os.path.basename(selected_folder)
        session_name = os.path.basename(os.path.dirname(selected_folder))
        output_filename = f"{ANIMAL_ID}_{session_name}_{folder_name}_no_labels.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Create the video with predictions but no ground truth
        video_path = create_labeled_video(selected_folder, frame_files, predictions, dummy_labels, output_path)
        
        if video_path:
            print(f"\nVideo created successfully!")
            print(f"Video saved to: {video_path}")
        else:
            print("Failed to create video")

if __name__ == "__main__":
    main() 