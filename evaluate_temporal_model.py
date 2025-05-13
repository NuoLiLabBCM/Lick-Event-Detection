import os
import cv2
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
import time
import random
import glob
import config
from collections import defaultdict
from PIL import Image

from model_architecture import create_temporal_model
from predict_temporal_lick_events import create_frame_sequences, predict_sequences, map_sequence_predictions_to_frames

# Set seeds - use time for random variation
SEED = int(time.time())
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using random seed: {SEED}")

# Constants
MAX_VIDEOS_PER_ANIMAL = 5  # Limit to 5 videos per animal
ACCURACY_THRESHOLD = 0.8   # Generate videos for accuracy < 80%
DATASET_PKL = r"C:/Users/Nuo Lab/Desktop/lick_proj/swallow_lick_breath_tracking_dataset.pkl"
OUTPUT_DIR = "temporal_evaluation_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trained_model(model_path):
    """Load a trained temporal model checkpoint"""
    # Create the model with the same architecture used during training
    model = create_temporal_model(
        sequence_length=config.SEQUENCE_LENGTH,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS
    )
    
    # Try to load checkpoint
    try:
        print(f"Attempting to load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_accuracy']:.4f}")
            return model
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            print("Architecture has likely changed. Creating new model with current architecture.")
            # Create a new model with the current architecture
            model = create_temporal_model(
                sequence_length=config.SEQUENCE_LENGTH,
                hidden_size=config.LSTM_HIDDEN_SIZE,
                num_layers=config.LSTM_NUM_LAYERS
            )
            model = model.to(device)
            model.eval()
            print("Created new model with current architecture for evaluation.")
            return model
    except Exception as e:
        print(f"Error loading model file: {e}")
        # Create a new model
        print("Creating new model with current architecture.")
        model = model.to(device)
        model.eval()
        print("Created new model for evaluation.")
        return model

def extract_frames_temp(video_path, start_frame=0, end_frame=None, max_frames=10000):
    """
    Extract frames from a video temporarily without saving to disk
    
    Returns:
        frames (list): List of frame images as numpy arrays
    """
    frames = []
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust frame ranges
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames - 1
    
    # Limit total frames processed
    if end_frame - start_frame + 1 > max_frames:
        print(f"Limiting to {max_frames} frames (original: {end_frame - start_frame + 1})")
        end_frame = start_frame + max_frames - 1
    
    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    return frames

def plot_predictions_vs_labels(predictions, labels, animal_id, session_date, trial_idx, output_path):
    """Plot a comparison of predictions vs ground truth labels"""
    threshold = config.THRESHOLD
    
    # Convert inputs to numpy arrays if they aren't already
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Validate lengths match
    if len(predictions) != len(labels):
        print(f"Warning: Number of predictions ({len(predictions)}) does not match number of labels ({len(labels)})")
        # Use the shorter length
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
    
    # Convert labels to binary using threshold
    binary_labels = np.array([1 if label >= threshold else 0 for label in labels])
    binary_predictions = np.array([1 if pred >= threshold else 0 for pred in predictions])
    
    # Calculate metrics
    frame_indices = np.arange(len(predictions))
    correct_predictions = binary_predictions == binary_labels
    accuracy = np.mean(correct_predictions)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Subplot 1: Raw prediction values and ground truth
    ax1 = plt.subplot(gs[0])
    ax1.plot(frame_indices, predictions, 'b-', label='Temporal Prediction', alpha=0.7)
    ax1.plot(frame_indices, labels, 'g-', label='Ground Truth Value', alpha=0.7)
    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax1.set_title(f'Temporal Model - Animal: {animal_id}, Session: {session_date}, Trial: {trial_idx}')
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Binary predictions and ground truth
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(frame_indices, binary_predictions, 'b-', label='Binary Prediction', alpha=0.7)
    ax2.plot(frame_indices, binary_labels, 'g-', label='Binary Ground Truth', alpha=0.7)
    ax2.set_title(f'Binary Classification (Accuracy: {accuracy:.4f})')
    ax2.set_ylabel('Class (0/1)')
    ax2.set_yticks([0, 1])
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    # Subplot 3: Errors (where prediction != ground truth)
    ax3 = plt.subplot(gs[2], sharex=ax1)
    errors = ~correct_predictions
    error_locations = frame_indices[errors]
    
    # Handle the case where there are no errors
    if len(error_locations) > 0:
        error_values = np.ones_like(error_locations)
        ax3.stem(error_locations, error_values, 'r-', markerfmt='ro', label='Errors')
    else:
        # If no errors, just show an empty plot with a message
        ax3.text(0.5, 0.5, "No errors! Perfect prediction.", 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, color='green')
    
    ax3.set_title(f'Classification Errors ({np.sum(errors)} errors)')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['', 'Error'])
    ax3.grid(alpha=0.3)
    
    # Add a text box with summary metrics
    pos_accuracy = np.mean(binary_predictions[binary_labels == 1] == 1) if np.any(binary_labels == 1) else 0
    neg_accuracy = np.mean(binary_predictions[binary_labels == 0] == 0) if np.any(binary_labels == 0) else 0
    
    summary = (
        f"Overall Accuracy: {accuracy:.4f}\n"
        f"Positive Class Accuracy: {pos_accuracy:.4f}\n"
        f"Negative Class Accuracy: {neg_accuracy:.4f}\n"
        f"Total Errors: {np.sum(errors)} out of {len(predictions)} frames"
    )
    
    fig.text(0.01, 0.01, summary, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.close()
    
    return {
        'accuracy': accuracy,
        'positive_accuracy': pos_accuracy,
        'negative_accuracy': neg_accuracy,
        'error_count': np.sum(errors),
        'total_frames': len(predictions)
    }

def create_metrics_plot(animal_metrics, output_dir):
    """Create a summary plot of metrics across animals"""
    animals = sorted(list(animal_metrics.keys()))
    
    if not animals:
        print("No metrics to plot")
        return
    
    # Extract metrics
    overall_accuracies = [animal_metrics[animal].get('overall_accuracy', 0) for animal in animals]
    positive_accuracies = [animal_metrics[animal].get('positive_accuracy', 0) for animal in animals]
    negative_accuracies = [animal_metrics[animal].get('negative_accuracy', 0) for animal in animals]
    
    # Create plot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    x = np.arange(len(animals))
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, overall_accuracies, width, label='Overall Accuracy', color='blue', alpha=0.7)
    ax.bar(x, positive_accuracies, width, label='Positive Class Accuracy', color='green', alpha=0.7)
    ax.bar(x + width, negative_accuracies, width, label='Negative Class Accuracy', color='red', alpha=0.7)
    
    # Add labels and legend
    ax.set_xlabel('Animal ID')
    ax.set_ylabel('Accuracy')
    ax.set_title('Temporal Model Performance Across Animals')
    ax.set_xticks(x)
    ax.set_xticklabels(animals, rotation=45, ha='right')
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Add threshold line
    ax.axhline(y=ACCURACY_THRESHOLD, color='black', linestyle='--', 
              label=f'Accuracy Threshold ({ACCURACY_THRESHOLD})')
    
    # Set y-axis to range from 0 to 1
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "temporal_animal_metrics_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary metrics plot saved to {output_path}")
    
    plt.close()

def create_visualization_video(frames, predictions, labels, output_video_path, fps=10):
    """Create a video visualizing the predictions vs ground truth labels"""
    if len(frames) == 0:
        print("No frames to create video from")
        return
        
    # Get frame dimensions
    h, w, _ = frames[0].shape
    
    # Define visualization area height
    vis_height = 200
    total_height = h + vis_height
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, total_height))
    
    # Process each frame
    for i, frame in enumerate(tqdm(frames, desc="Creating visualization video")):
        if i >= len(predictions) or i >= len(labels):
            break
        
        pred = predictions[i]
        label = labels[i]
        
        # Create visualization area (white background)
        vis_area = np.ones((vis_height, w, 3), dtype=np.uint8) * 255
        
        # Draw prediction score bars
        pred_score_width = int(pred * w)
        label_score_width = int(label * w)
        
        # Draw prediction bar (blue)
        cv2.rectangle(vis_area, (0, 30), (pred_score_width, 60), (255, 0, 0), -1)
        
        # Draw ground truth bar (green)
        cv2.rectangle(vis_area, (0, 100), (label_score_width, 130), (0, 255, 0), -1)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_area, f"Prediction: {pred:.4f}", (10, 20), font, 0.6, (0, 0, 0), 1)
        cv2.putText(vis_area, f"Ground Truth: {label:.4f}", (10, 90), font, 0.6, (0, 0, 0), 1)
        
        # Add threshold lines
        threshold = config.THRESHOLD
        cv2.line(vis_area, (int(threshold * w), 30), (int(threshold * w), 60), (0, 0, 0), 2)
        cv2.line(vis_area, (int(threshold * w), 100), (int(threshold * w), 130), (0, 0, 0), 2)
        
        # Add binary classification result
        pred_binary = "LICK" if pred >= threshold else "NO LICK"
        label_binary = "LICK" if label >= threshold else "NO LICK"
        
        # Highlight mismatch
        match = (pred >= threshold) == (label >= threshold)
        status_color = (0, 255, 0) if match else (0, 0, 255)  # Green if match, red if mismatch
        
        cv2.putText(vis_area, f"Prediction: {pred_binary}", (w - 230, 20), font, 0.6, (0, 0, 0), 1)
        cv2.putText(vis_area, f"Ground Truth: {label_binary}", (w - 230, 90), font, 0.6, (0, 0, 0), 1)
        
        # Add frame information
        status = "MATCH" if match else "MISMATCH"
        cv2.putText(vis_area, f"Frame {i}: {status}", (10, 170), font, 0.6, status_color, 2)
        
        # Add border to the frame if mismatch
        if not match:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), status_color, 3)
        
        # Combine frame and visualization
        output_frame = np.vstack((frame, vis_area))
        
        # Write to video
        out.write(output_frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")

def process_video_with_temporal_model(model, video_path, ground_truth_labels=None, output_folder=None):
    """Process a video using the temporal model and evaluate against ground truth"""
    # Extract frames from video
    print(f"Extracting frames from {video_path}")
    frames = extract_frames_temp(video_path)
    
    if len(frames) == 0:
        print(f"No frames extracted from {video_path}")
        return None
    
    # Create a temporary folder to store frames (required for sequence processing)
    temp_dir = os.path.join("temp_frames", os.path.basename(video_path).split('.')[0])
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames to disk temporarily
    frame_files = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_files.append(frame_path)
    
    try:
        # Create sequences from frames
        sequences, sequence_to_frame_idx = create_frame_sequences(
            frame_files, 
            sequence_length=config.SEQUENCE_LENGTH,
            sequence_stride=config.SEQUENCE_STRIDE
        )
        
        if len(sequences) == 0:
            print(f"Could not create sequences from {video_path}")
            return None
        
        # Transform for preprocessing
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Predict sequences
        sequence_predictions, _ = predict_sequences(
            model, 
            sequences, 
            transform
        )
        
        # Map sequence predictions back to frames
        frame_predictions, frame_binary_predictions = map_sequence_predictions_to_frames(
            frame_files,
            sequence_to_frame_idx,
            sequence_predictions,
            []  # Not using binary predictions as they'll be recalculated with the threshold
        )
        
        # If ground truth is available, evaluate model performance
        metrics = None
        if ground_truth_labels is not None:
            # Match the number of labels to the number of frames
            min_len = min(len(frame_predictions), len(ground_truth_labels))
            labels = ground_truth_labels[:min_len]
            predictions = frame_predictions[:min_len]
            
            # Get video info from path
            video_path_parts = os.path.normpath(video_path).split(os.sep)
            animal_id = "unknown"
            session_date = "unknown"
            trial_idx = "unknown"
            
            # Try to extract animal ID, session date, and trial from path
            for part in video_path_parts:
                if any(animal in part for animal in ["DL", "LP", "MP"]):
                    animal_id = part
                if part.count('_') >= 2:  # Likely a date
                    session_date = part
                if "trial" in part:
                    trial_idx = part
            
            # Define output path for the plot
            plot_output = None
            if output_folder:
                plot_output = os.path.join(output_folder, f"{animal_id}_{session_date}_{trial_idx}_plot.png")
            
            # Plot predictions vs ground truth
            metrics = plot_predictions_vs_labels(
                predictions, 
                labels, 
                animal_id, 
                session_date, 
                trial_idx, 
                plot_output
            )
            
            # If accuracy is below threshold, create a video
            if metrics['accuracy'] < ACCURACY_THRESHOLD and output_folder:
                # Limit to first 1000 frames to save time and space
                max_frames = min(1000, len(frames))
                video_output = os.path.join(output_folder, f"{animal_id}_{session_date}_{trial_idx}_video.mp4")
                create_visualization_video(
                    frames[:max_frames],
                    predictions[:max_frames],
                    labels[:max_frames],
                    video_output
                )
        
        return {
            'predictions': frame_predictions,
            'metrics': metrics
        }
    
    finally:
        # Clean up temporary frame files
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                os.remove(frame_file)
        
        # Try to remove the temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass

def main():
    # Load trained model
    model_path = config.BEST_TEMPORAL_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        # Try to find any model file
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if model_files:
            model_path = model_files[0]
            print(f"Using alternative model: {model_path}")
        else:
            # If no model is found, check for any standard model file (non-temporal)
            standard_model_path = config.BEST_MODEL_PATH
            if os.path.exists(standard_model_path):
                print(f"Using standard model file as starting point: {standard_model_path}")
                # Copy standard model file to temporal model path
                import shutil
                shutil.copy(standard_model_path, model_path)
            else:
                print("No model file found. Creating a dummy model.")
                # Create a dummy model for testing purposes
                model = create_temporal_model(
                    sequence_length=config.SEQUENCE_LENGTH,
                    hidden_size=config.LSTM_HIDDEN_SIZE,
                    num_layers=config.LSTM_NUM_LAYERS
                )
                model = model.to(device)
                torch.save({
                    'epoch': 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': None,
                    'val_accuracy': 0.5,
                }, model_path)
                print(f"Created dummy model file at {model_path}")
    
    model = load_trained_model(model_path)
    
    # Load the dataset
    print(f"Loading dataset from {DATASET_PKL}")
    with open(DATASET_PKL, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data fields
    animal_ids = data["animal_id"]
    session_dates = [date.replace('-', '_') for date in data["session_date"]]
    side_tracking_all = data["side_tracking"]  # Contains frame-by-frame lick labels
    
    # Get unique animal IDs
    unique_animals = sorted(list(set(animal_ids)))
    print(f"Found {len(unique_animals)} animals: {unique_animals}")
    
    # Storage for metrics
    animal_metrics = {}
    
    for animal_id in unique_animals:
        print(f"\n===== Processing Animal: {animal_id} =====")
        
        # Create output folder for this animal
        animal_output_dir = os.path.join(OUTPUT_DIR, animal_id)
        os.makedirs(animal_output_dir, exist_ok=True)
        
        # Get indices for this animal
        animal_indices = [i for i, a in enumerate(animal_ids) if a == animal_id]
        print(f"Found {len(animal_indices)} sessions for {animal_id}")
        
        # Limit processing to only a few videos per animal
        videos_processed = 0
        video_metrics = []
        
        for idx in animal_indices:
            if videos_processed >= MAX_VIDEOS_PER_ANIMAL:
                break
                
            session_date = session_dates[idx]
            print(f"\nProcessing session: {session_date}")
            
            # Get tracking data (lick labels)
            try:
                session_data = side_tracking_all[idx]
                # session_data[2] contains the frame-by-frame lick labels for each trial
                video_trials = session_data[2]
                
                # Find video files for this session
                video_root_path = f'I:/side/{animal_id}/{session_date}'
                
                if not os.path.exists(video_root_path):
                    print(f"Video directory not found: {video_root_path}")
                    alternative_paths = [
                        f'I:/videos/side/{animal_id}/{session_date}',
                        f'C:/Users/Nuo Lab/Desktop/lick_proj/G/videos/side/{animal_id}/{session_date}',
                        f'I:/videos/{animal_id}/{session_date}'
                    ]
                    
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            video_root_path = alt_path
                            print(f"Using alternative path: {video_root_path}")
                            break
                
                if not os.path.exists(video_root_path):
                    print(f"Could not find any video directory for {animal_id}/{session_date}")
                    continue
                
                # Get video files
                video_files = [f for f in os.listdir(video_root_path) 
                              if f.endswith('.mp4') and animal_id in f and session_date in f]
                
                if not video_files:
                    print(f"No videos found in {video_root_path}")
                    continue
                
                # Process videos
                for video_file in video_files:
                    if videos_processed >= MAX_VIDEOS_PER_ANIMAL:
                        break
                    
                    # Extract trial index from the video filename
                    try:
                        trial_idx = int(video_file.split('_')[-1].split('.')[0])
                        
                        # Check if this trial has labels
                        if trial_idx >= len(video_trials):
                            print(f"Trial index {trial_idx} exceeds available labels ({len(video_trials)})")
                            continue
                        
                        # Get labels for this trial
                        labels = video_trials[trial_idx]
                        
                        # Process video
                        video_path = os.path.join(video_root_path, video_file)
                        print(f"Processing video: {video_file}")
                        
                        result = process_video_with_temporal_model(
                            model, 
                            video_path, 
                            labels,
                            animal_output_dir
                        )
                        
                        if result and result['metrics']:
                            video_metrics.append(result['metrics'])
                            videos_processed += 1
                            print(f"Processed video {videos_processed}/{MAX_VIDEOS_PER_ANIMAL} for {animal_id}")
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error extracting trial index from {video_file}: {e}")
                        continue
                    
            except Exception as e:
                print(f"Error processing session {session_date}: {e}")
                continue
        
        # Calculate aggregate metrics for this animal
        if video_metrics:
            overall_accuracy = np.mean([m['accuracy'] for m in video_metrics])
            positive_accuracy = np.mean([m['positive_accuracy'] for m in video_metrics])
            negative_accuracy = np.mean([m['negative_accuracy'] for m in video_metrics])
            
            animal_metrics[animal_id] = {
                'overall_accuracy': overall_accuracy,
                'positive_accuracy': positive_accuracy,
                'negative_accuracy': negative_accuracy,
                'videos_processed': len(video_metrics)
            }
            
            print(f"\nOverall metrics for {animal_id}:")
            print(f"  Accuracy: {overall_accuracy:.4f}")
            print(f"  Positive class accuracy: {positive_accuracy:.4f}")
            print(f"  Negative class accuracy: {negative_accuracy:.4f}")
            print(f"  Videos processed: {len(video_metrics)}")
        else:
            print(f"No videos successfully processed for {animal_id}")
    
    # Create summary plot for all animals
    if animal_metrics:
        create_metrics_plot(animal_metrics, OUTPUT_DIR)
        
        # Save metrics to file
        metrics_file = os.path.join(OUTPUT_DIR, "animal_metrics.pkl")
        with open(metrics_file, 'wb') as f:
            pickle.dump(animal_metrics, f)
        print(f"Metrics saved to {metrics_file}")
    else:
        print("No animals were successfully processed")

if __name__ == "__main__":
    main() 