#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the temporal model performance on all animals' data.
For each animal, randomly select 5 videos, run the model, and create comparison plots.
If accuracy is below threshold, generate prediction videos as well.
"""

import os
import pickle
import numpy as np
import torch
import random
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def get_labels_path(animal_id, dataset_type="test"):
    """Get the path to the labels pickle file"""
    return f"C:/Users/Nuo Lab/Desktop/lick/withframenew/datatags_{VIEW}_corrected/{dataset_type}_labels_lick_{animal_id}.pkl"

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

def load_animal_labels(animal_id, data=None):
    """Load labels for an animal directly from the main dataset or from separate pkl files"""
    combined_labels = {}
    
    # First try to load from the main dataset (preferred method)
    if data is not None:
        animal_indices = [i for i, aid in enumerate(data["animal_id"]) if aid == animal_id]
        
        if animal_indices:
            print(f"Found {len(animal_indices)} sessions for {animal_id} in main dataset")
            
            for idx in animal_indices:
                session_date = data["session_date"][idx].replace('-', '_')
                side_tracking = data["side_tracking"][idx]
                
                # Skip if there's not enough session data
                if len(side_tracking) < 3:
                    continue
                
                # Extract trial data which contains frame-by-frame lick labels
                video_trials = side_tracking[2]
                
                # Create the folder path pattern that would match this animal's sessions
                frame_base_path = f"I:/frames/side/{animal_id}/{session_date}"
                
                # For each trial, create a folder path and associate with its labels
                for trial_idx, trial_labels in enumerate(video_trials):
                    if len(trial_labels) > 0:  # Skip empty label arrays
                        folder_path = f"{frame_base_path}/trial_{trial_idx}"
                        
                        # Check if the folder exists
                        if os.path.exists(folder_path) and glob.glob(os.path.join(folder_path, "frame_*.jpg")):
                            combined_labels[folder_path] = np.array(trial_labels)
            
            if combined_labels:
                return combined_labels
    
    # Fallback: Try to load from separate pkl files
    try:
        test_labels_path = get_labels_path(animal_id, "test")
        with open(test_labels_path, 'rb') as f:
            test_labels = pickle.load(f)
            combined_labels.update(test_labels)
            print(f"Loaded {len(test_labels)} test videos for {animal_id} from pkl")
    except Exception as e:
        print(f"Error loading test labels for {animal_id}: {e}")
    
    try:
        train_labels_path = get_labels_path(animal_id, "train")
        with open(train_labels_path, 'rb') as f:
            train_labels = pickle.load(f)
            combined_labels.update(train_labels)
            print(f"Loaded {len(train_labels)} train videos for {animal_id} from pkl")
    except Exception as e:
        print(f"Error loading train labels for {animal_id}: {e}")
    
    return combined_labels

def calculate_metrics(y_true, y_pred, display=True):
    """Calculate and return performance metrics"""
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
    """Create and save a plot comparing predictions and ground truth"""
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

def evaluate_animal(animal_id, model, dataset=None):
    """Evaluate the model on videos from a specific animal"""
    print(f"\n{'='*80}")
    print(f"Evaluating Animal: {animal_id}")
    print(f"{'='*80}")
    
    # Load all labels for this animal
    all_labels = load_animal_labels(animal_id, dataset)
    
    if not all_labels:
        print(f"No labels found for animal {animal_id}, skipping.")
        return None
    
    # Filter to keep only folders that exist and have frames
    valid_folders = []
    for folder in all_labels.keys():
        if os.path.exists(folder) and len(glob.glob(os.path.join(folder, "frame_*.jpg"))) > 0:
            valid_folders.append(folder)
    
    if not valid_folders:
        print(f"No valid frame folders found for animal {animal_id}, skipping.")
        return None
    
    # Select random videos for evaluation (or all if there are fewer than requested)
    num_videos = min(NUM_VIDEOS_PER_ANIMAL, len(valid_folders))
    selected_folders = random.sample(valid_folders, num_videos)
    
    # Prepare results data
    animal_results = []
    
    # Process each selected folder
    for folder in selected_folders:
        # Extract session date and trial number from folder path
        folder_parts = os.path.normpath(folder).split(os.sep)
        session_date = "unknown"
        trial_num = "unknown"
        
        for i, part in enumerate(folder_parts):
            if part.startswith("trial_"):
                trial_num = part
                if i > 0:
                    potential_date = folder_parts[i-1]
                    if "_" in potential_date:
                        session_date = potential_date
        
        video_name = f"{animal_id}_{session_date}_{trial_num}"
        print(f"\nProcessing video: {video_name}")
        
        # Get the ground truth labels
        labels = all_labels[folder]
        
        # Get model predictions for the frames
        frame_files, predictions = get_frame_predictions(model, folder)
        
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
            animal_id, session_date, trial_num, metrics
        )
        
        # Create labeled video if accuracy is below threshold
        video_path = None
        if metrics['accuracy'] < ACCURACY_THRESHOLD or metrics['positive_accuracy'] < POSITIVE_ACCURACY_THRESHOLD:
            print(f"Accuracy below threshold, creating labeled video...")
            output_filename = f"{video_name}_predictions.mp4"
            output_path = os.path.join(OUTPUT_DIR, "videos", output_filename)
            
            # Create the labeled video
            video_path = create_labeled_video(folder, frame_files, predictions, labels, output_path)
            if video_path:
                print(f"Created video: {video_path}")
        
        # Store results
        result = {
            'animal_id': animal_id,
            'session_date': session_date,
            'trial_num': trial_num,
            'video_name': video_name,
            'folder': folder,
            'plot_path': plot_path,
            'video_path': video_path,
            'num_frames': len(frame_files),
            'num_positive_frames': sum((np.array(labels) >= THRESHOLD).astype(int)),
            **metrics  # Include all metrics
        }
        
        animal_results.append(result)
    
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
    
    # Evaluate each animal
    for animal_id in animal_ids:
        animal_results = evaluate_animal(animal_id, model, dataset)
        if animal_results:
            all_results.extend(animal_results)
    
    # Create a summary dataframe and save it
    if all_results:
        df = pd.DataFrame(all_results)
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
        animal_summary.to_csv(animal_summary_path)
        print(f"Saved animal summary to {animal_summary_path}")
    else:
        print("No results collected. Check if animal labels are available.")

if __name__ == "__main__":
    main() 