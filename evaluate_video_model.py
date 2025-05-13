#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the temporal model performance on all animals' data.
For each animal, randomly select 5 videos, extract frames, run the model, 
and create comparison plots. If accuracy is below threshold, generate prediction videos as well.

This script works directly with video files rather than pre-extracted frames.
"""

import os
import pickle
import numpy as np
import torch
import random
import glob
import cv2
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tempfile
import time

# Import necessary functions from video.py
from video import (
    get_frame_predictions, load_trained_model, device,
    create_labeled_video, MODEL_PATH, THRESHOLD, VIEW
)

# Import config
import config

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Constants
ACCURACY_THRESHOLD = 80.0  # Minimum required accuracy (%)
POSITIVE_ACCURACY_THRESHOLD = 50.0  # Minimum required positive class accuracy (%)
NUM_VIDEOS_PER_ANIMAL = 5  # Number of videos to sample for each animal
OUTPUT_DIR = "evaluation_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "videos"), exist_ok=True)

# Create a temporary directory for extracted frames
TEMP_FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_frames")
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)

def extract_frames(video_path, output_folder, max_frames=None):
    """
    Extract frames from a video file
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Directory to save the extracted frames
        max_frames (int, optional): Maximum number of frames to extract. If None, extract all frames.
        
    Returns:
        int: Number of frames extracted
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"Extracting {total_frames} frames from video at {fps} fps")
    
    # Extract frames
    frame_count = 0
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save the frame
            frame_file = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return frame_count

def load_main_dataset():
    """Load the main dataset pickle file with all animal data"""
    try:
        dataset_pkl = r"C:/Users/Nuo Lab/Desktop/lick_proj/swallow_lick_breath_tracking_dataset.pkl"
        with open(dataset_pkl, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading main dataset: {e}")
        return None

def get_video_paths_and_labels(animal_id, data):
    """
    Get video paths and corresponding labels for a specific animal.
    
    Args:
        animal_id (str): Animal ID
        data (dict): Dataset dictionary
        
    Returns:
        dict: Dictionary mapping video paths to their trial index and labels
    """
    if data is None:
        print(f"No dataset provided for animal {animal_id}")
        return {}
    
    videos_with_labels = {}
    
    # Find indices for this animal
    animal_indices = [i for i, aid in enumerate(data["animal_id"]) if aid == animal_id]
    
    if not animal_indices:
        print(f"No data found for animal {animal_id}")
        return {}
    
    print(f"Found {len(animal_indices)} sessions for {animal_id}")
    
    for idx in animal_indices:
        session_date = data["session_date"][idx].replace('-', '_')
        side_tracking = data["side_tracking"][idx]
        
        # Skip if there's not enough session data
        if len(side_tracking) < 3:
            print(f"Skipping session {session_date} for {animal_id} - insufficient data")
            continue
        
        # Extract trial data which contains frame-by-frame lick labels
        video_trials = side_tracking[2]
        
        # Check if videos exist for this animal and session
        video_dir = f"I:/side/{animal_id}/{session_date}"
        if not os.path.exists(video_dir):
            print(f"Video directory not found: {video_dir}")
            continue
        
        video_files = [f for f in os.listdir(video_dir) 
                      if f.endswith('.mp4') and animal_id in f and session_date in f]
        
        for video_file in video_files:
            try:
                # Extract trial index from filename
                trial_idx = int(video_file.split('_')[-1].split('.')[0])
                
                # Check if we have labels for this trial
                if trial_idx < len(video_trials) and len(video_trials[trial_idx]) > 0:
                    video_path = os.path.join(video_dir, video_file)
                    videos_with_labels[video_path] = {
                        'trial_idx': trial_idx,
                        'labels': video_trials[trial_idx],
                        'session_date': session_date
                    }
            except (ValueError, IndexError) as e:
                print(f"Could not extract trial index from {video_file}: {e}")
                continue
    
    print(f"Found {len(videos_with_labels)} videos with labels for animal {animal_id}")
    return videos_with_labels

def calculate_metrics(y_true, y_pred, display=True):
    """
    Calculate and return performance metrics
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        display (bool): Whether to print metrics
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to binary predictions
    y_true_binary = (np.array(y_true) >= THRESHOLD).astype(int)
    y_pred_binary = (np.array(y_pred) >= THRESHOLD).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true_binary, y_pred_binary) * 100
    
    # Calculate per-class metrics (need to handle empty arrays)
    if sum(y_true_binary) > 0:
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0) * 100
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0) * 100
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0) * 100
    else:
        prec, rec, f1 = 0, 0, 0
    
    # Calculate custom positive accuracy (what percentage of positive frames are correctly detected)
    positive_indices = np.where(y_true_binary == 1)[0]
    if len(positive_indices) > 0:
        positive_accuracy = 100 * np.mean(y_pred_binary[positive_indices] == y_true_binary[positive_indices])
    else:
        positive_accuracy = 0
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    
    if display:
        print(f"Accuracy: {acc:.2f}%")
        print(f"Precision: {prec:.2f}%")
        print(f"Recall: {rec:.2f}%")
        print(f"F1 Score: {f1:.2f}%")
        print(f"Positive Accuracy: {positive_accuracy:.2f}%")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return {
        'accuracy': acc,
        'precision': prec, 
        'recall': rec,
        'f1_score': f1,
        'positive_accuracy': positive_accuracy,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

def plot_predictions(frame_files, predictions, labels, video_name, animal_id, session_date, trial_num, metrics):
    """
    Create and save a plot comparing predictions and ground truth
    
    Args:
        frame_files (list): List of frame file paths
        predictions (array-like): Model predictions
        labels (array-like): Ground truth labels
        video_name (str): Name of the video
        animal_id (str): Animal ID
        session_date (str): Session date
        trial_num (str): Trial number
        metrics (dict): Metrics dictionary
        
    Returns:
        str: Path to the saved plot
    """
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [5, 1, 1]})
    
    # Plot raw predictions and ground truth
    x = np.arange(len(predictions))
    ax1.plot(x, predictions, 'b-', alpha=0.7, label='Model Predictions')
    ax1.plot(x, labels, 'g-', alpha=0.7, label='Ground Truth')
    ax1.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD})')
    
    # Highlight areas where binary prediction doesn't match ground truth
    y_true_binary = (np.array(labels) >= THRESHOLD).astype(int)
    y_pred_binary = (np.array(predictions) >= THRESHOLD).astype(int)
    
    # Find regions where predictions don't match
    mismatch = y_true_binary != y_pred_binary
    
    # Highlight false positives
    fp = np.logical_and(mismatch, y_pred_binary == 1)
    if np.any(fp):
        ax1.fill_between(x, 0, 1, where=fp, color='r', alpha=0.3, label='False Positive')
    
    # Highlight false negatives
    fn = np.logical_and(mismatch, y_pred_binary == 0)
    if np.any(fn):
        ax1.fill_between(x, 0, 1, where=fn, color='orange', alpha=0.3, label='False Negative')
    
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, len(predictions))
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title(f'{animal_id} - {session_date} - {trial_num} (Accuracy: {metrics["accuracy"]:.1f}%, Pos. Acc: {metrics["positive_accuracy"]:.1f}%)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot binary predictions
    ax2.plot(x, y_pred_binary, 'b-', drawstyle='steps-post', label='Binary Prediction')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(0, len(predictions))
    ax2.set_ylabel('Prediction')
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Plot ground truth binary
    ax3.plot(x, y_true_binary, 'g-', drawstyle='steps-post', label='Binary Ground Truth')
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xlim(0, len(predictions))
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Ground Truth')
    ax3.set_yticks([0, 1])
    ax3.grid(True, alpha=0.3)
    
    # Add metrics as text annotation
    metrics_text = (
        f"Accuracy: {metrics['accuracy']:.1f}%\n"
        f"Precision: {metrics['precision']:.1f}%\n"
        f"Recall: {metrics['recall']:.1f}%\n"
        f"F1 Score: {metrics['f1_score']:.1f}%\n"
        f"Positive Accuracy: {metrics['positive_accuracy']:.1f}%\n"
        f"TP: {metrics['tp']}, FP: {metrics['fp']}\n"
        f"TN: {metrics['tn']}, FN: {metrics['fn']}"
    )
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plots", f"{video_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return plot_path

def evaluate_animal_videos(animal_id, model, dataset):
    """
    Evaluate the model on videos from a specific animal.
    
    Args:
        animal_id (str): Animal ID
        model: Trained model
        dataset (dict): Dataset dictionary
        
    Returns:
        list: List of results dictionaries
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Animal: {animal_id}")
    print(f"{'='*80}")
    
    # Get video paths and labels for this animal
    videos_with_labels = get_video_paths_and_labels(animal_id, dataset)
    
    if not videos_with_labels:
        print(f"No videos with labels found for animal {animal_id}, skipping.")
        return None
    
    # Select random videos for evaluation (or all if there are fewer than requested)
    video_paths = list(videos_with_labels.keys())
    num_videos = min(NUM_VIDEOS_PER_ANIMAL, len(video_paths))
    selected_videos = random.sample(video_paths, num_videos)
    
    # Prepare results data
    animal_results = []
    
    # Process each selected video
    for video_path in selected_videos:
        video_info = videos_with_labels[video_path]
        session_date = video_info['session_date']
        trial_idx = video_info['trial_idx']
        labels = video_info['labels']
        
        # Create a video name for display/saving
        video_filename = os.path.basename(video_path)
        video_name = f"{animal_id}_{session_date}_trial_{trial_idx}"
        print(f"\nProcessing video: {video_name}")
        
        # Create a unique temporary directory for this video's frames
        temp_dir = os.path.join(TEMP_FRAMES_DIR, f"{animal_id}_{session_date}_trial_{trial_idx}_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Extract frames from the video
            num_frames = extract_frames(video_path, temp_dir)
            
            if num_frames == 0:
                print(f"Failed to extract frames from {video_path}")
                continue
            
            # Check if we have enough labels
            if len(labels) < num_frames:
                print(f"Warning: Only {len(labels)} labels available but {num_frames} frames extracted")
                # Resize labels if needed
                if len(labels) > 0:
                    # Repeat the last label value to match frame count
                    labels = np.pad(labels, (0, num_frames - len(labels)), mode='edge')
                else:
                    print(f"No valid labels for {video_path}, skipping")
                    continue
            elif len(labels) > num_frames:
                # Truncate labels if we have more labels than frames
                labels = labels[:num_frames]
            
            # Get model predictions for the frames
            frame_files, predictions = get_frame_predictions(model, temp_dir)
            
            # Make sure lengths match
            min_len = min(len(frame_files), len(predictions), len(labels))
            frame_files = frame_files[:min_len]
            predictions = predictions[:min_len]
            labels = labels[:min_len]
            
            # Calculate metrics
            metrics = calculate_metrics(labels, predictions, display=True)
            
            # Create and save plot
            plot_path = plot_predictions(
                frame_files, predictions, labels, video_name,
                animal_id, session_date, str(trial_idx), metrics
            )
            
            # Create labeled video if accuracy is below threshold
            video_output_path = None
            if metrics['accuracy'] < ACCURACY_THRESHOLD or metrics['positive_accuracy'] < POSITIVE_ACCURACY_THRESHOLD:
                print(f"Accuracy below threshold, creating labeled video...")
                output_filename = f"{video_name}_predictions.mp4"
                video_output_path = os.path.join(OUTPUT_DIR, "videos", output_filename)
                
                # Create the labeled video
                video_path = create_labeled_video(temp_dir, frame_files, predictions, labels, video_output_path)
                if video_path:
                    print(f"Created video: {video_path}")
            
            # Store results
            result = {
                'animal_id': animal_id,
                'session_date': session_date,
                'trial_num': trial_idx,
                'video_name': video_name,
                'video_path': video_path,
                'output_video': video_output_path,
                'plot_path': plot_path,
                'num_frames': len(frame_files),
                'num_positive_frames': sum((np.array(labels) >= THRESHOLD).astype(int)),
                **metrics  # Include all metrics
            }
            
            animal_results.append(result)
            
        finally:
            # Clean up temporary frame directory
            print(f"Cleaning up temporary frames directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")
    
    return animal_results

def main():
    print("Loading model...")
    model = load_trained_model(MODEL_PATH)
    
    # Load main dataset
    print("Loading main dataset...")
    dataset = load_main_dataset()
    
    # Get all animal IDs
    if dataset is not None:
        animal_ids = list(set(dataset["animal_id"]))
        print(f"Found {len(animal_ids)} animals in dataset: {', '.join(animal_ids[:5])}...")
    else:
        # Fallback to hardcoded animal IDs if needed
        animal_ids = ["DL001", "DL002", "DL005", "DL006", "DL007", "DL012", "DL013", 
                      "DL024", "DL025", "DL026", "DL027", "DL028", "DL029"]
        print(f"Using hardcoded list of {len(animal_ids)} animals")
    
    # Initialize results
    all_results = []
    
    # Process a specific animal or all animals
    target_animal = None  # Set to None to process all animals, or specify an animal ID (e.g., "DL005")
    
    # Filter animal IDs if a target is specified
    if target_animal:
        animal_ids = [aid for aid in animal_ids if aid == target_animal]
        print(f"Processing only animal {target_animal}")
    
    # Evaluate each animal
    for animal_id in animal_ids:
        animal_results = evaluate_animal_videos(animal_id, model, dataset)
        if animal_results:
            all_results.extend(animal_results)
    
    # Create a summary dataframe and save it
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Ensure the evaluation_results directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save full results
        summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nSaved evaluation summary to {summary_path}")
        
        # Print overall metrics
        print("\nOverall Metrics:")
        print(f"Average Accuracy: {df['accuracy'].mean():.2f}%")
        print(f"Average Precision: {df['precision'].mean():.2f}%")
        print(f"Average Recall: {df['recall'].mean():.2f}%")
        print(f"Average F1 Score: {df['f1_score'].mean():.2f}%")
        print(f"Average Positive Accuracy: {df['positive_accuracy'].mean():.2f}%")
        
        # Create a summary per animal
        print("\nPer-Animal Summary:")
        animal_summary = df.groupby('animal_id').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean',
            'positive_accuracy': 'mean',
            'tp': 'sum',
            'fp': 'sum',
            'tn': 'sum',
            'fn': 'sum'
        })
        
        print(animal_summary)
        
        # Save animal summary
        animal_summary_path = os.path.join(OUTPUT_DIR, "animal_summary.csv")
        try:
            animal_summary.to_csv(animal_summary_path)
            print(f"Saved animal summary to {animal_summary_path}")
        except Exception as e:
            print(f"Error saving animal summary: {e}")
    else:
        print("No results collected. Check if animal labels are available.")
    
    # Clean up the temporary frames directory
    try:
        if os.path.exists(TEMP_FRAMES_DIR) and os.path.isdir(TEMP_FRAMES_DIR):
            # Only remove subdirectories, keep the main directory
            for subdir in os.listdir(TEMP_FRAMES_DIR):
                subdir_path = os.path.join(TEMP_FRAMES_DIR, subdir)
                if os.path.isdir(subdir_path):
                    shutil.rmtree(subdir_path)
            print(f"Cleaned up temporary frames directory: {TEMP_FRAMES_DIR}")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory: {e}")

if __name__ == "__main__":
    main() 