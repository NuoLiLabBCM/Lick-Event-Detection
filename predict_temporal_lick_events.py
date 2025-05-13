import argparse
import os
import glob
import numpy as np
import torch
import cv2
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
import config  # Import the centralized configuration
from model_architecture import create_temporal_model  # Import the model function

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_trained_model(model_path):
    """Load a trained temporal model checkpoint"""
    # Create the model with the same architecture used during training
    model = create_temporal_model(
        sequence_length=config.SEQUENCE_LENGTH,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print model info with safe access to checkpoint keys
    epoch = checkpoint.get('epoch', 'unknown')
    val_accuracy = checkpoint.get('val_accuracy', checkpoint.get('accuracy', 0.0))
    print(f"Loaded model from epoch {epoch} with validation accuracy: {val_accuracy:.4f}")
    
    return model

def create_frame_sequences(frame_files, sequence_length=10, sequence_stride=1):
    """
    Create overlapping sequences of frames from a list of frame files.
    
    Args:
        frame_files: List of frame file paths
        sequence_length: Number of frames in each sequence
        sequence_stride: Step size between sequences
        
    Returns:
        List of sequences, where each sequence is a list of frame paths
        Dictionary mapping sequence index to original frame index (for the last frame in the sequence)
    """
    # Always use step size of 1 for smoother predictions
    sequence_stride = 1  # Override the input stride
    
    sequences = []
    sequence_to_frame_idx = {}
    
    # Skip if not enough frames
    if len(frame_files) < sequence_length:
        print(f"Warning: Only {len(frame_files)} frames available, which is less than the required sequence length {sequence_length}")
        return sequences, sequence_to_frame_idx
    
    # Create sequences
    for i in range(0, len(frame_files) - sequence_length + 1, sequence_stride):
        sequence = frame_files[i:i+sequence_length]
        sequences.append(sequence)
        # Map the sequence index to the original frame index (for the last frame)
        sequence_to_frame_idx[len(sequences)-1] = i + sequence_length - 1
    
    return sequences, sequence_to_frame_idx

def predict_sequences(model, sequences, transform, batch_size=32, threshold=None):
    """
    Run inference on sequences of frames
    
    Args:
        model: The trained temporal model
        sequences: List of sequences, where each sequence is a list of frame paths
        transform: PyTorch transform to apply to images
        batch_size: Batch size for inference
        threshold: Threshold for binary classification
        
    Returns:
        sequence_predictions: Continuous predictions for each sequence
        sequence_binary_predictions: Binary predictions for each sequence
    """
    # Use threshold from config if not specified
    threshold = threshold if threshold is not None else config.THRESHOLD
    
    if not sequences:
        print("No sequences to predict")
        return [], []
    
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
        
        # Set model to evaluation mode to trigger smoothing
        model.eval()
        
        # Run inference
        with torch.no_grad():
            batch_preds = model(batch_tensor).squeeze().cpu().numpy()
            
            # Apply sigmoid since it's no longer in the model
            batch_preds = 1.0 / (1.0 + np.exp(-batch_preds))
            
            # Handle the case where there's only one prediction
            if not isinstance(batch_preds, np.ndarray):
                batch_preds = np.array([batch_preds])
                
            sequence_predictions.extend(batch_preds)
    
    # Convert to binary predictions
    sequence_binary_predictions = [1 if pred >= threshold else 0 for pred in sequence_predictions]
    
    # Calculate statistics
    positive_count = sum(sequence_binary_predictions)
    total_count = len(sequence_binary_predictions)
    positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Total sequences: {total_count}")
    print(f"Positive predictions: {positive_count} ({positive_percentage:.2f}%)")
    
    return sequence_predictions, sequence_binary_predictions

def map_sequence_predictions_to_frames(frame_files, sequence_to_frame_idx, 
                                      sequence_predictions, sequence_binary_predictions):
    """
    Map sequence predictions back to individual frames.
    For frames that appear in multiple sequences, take the maximum prediction.
    
    Args:
        frame_files: List of all frame files
        sequence_to_frame_idx: Dictionary mapping sequence index to original frame index
        sequence_predictions: Continuous predictions for each sequence
        sequence_binary_predictions: Binary predictions for each sequence
        
    Returns:
        frame_predictions: Continuous predictions for each frame
        frame_binary_predictions: Binary predictions for each frame
    """
    # Initialize arrays for frame predictions
    frame_predictions = np.zeros(len(frame_files))
    frame_coverage = np.zeros(len(frame_files))  # Count how many sequences cover each frame
    
    # Create reference to all sequence indices to reconstruct sequence-to-frame mapping
    seq_length = config.SEQUENCE_LENGTH
    seq_stride = config.SEQUENCE_STRIDE
    
    # Map sequence predictions to frames with weighted contribution
    for seq_idx, frame_idx in sequence_to_frame_idx.items():
        # Calculate the range of frames in this sequence
        seq_start_idx = frame_idx - seq_length + 1
        
        # Weight predictions based on position in sequence (higher weight to middle frames)
        for offset in range(seq_length):
            # Skip frames that are outside our valid range
            if seq_start_idx + offset < 0 or seq_start_idx + offset >= len(frame_files):
                continue
                
            # Calculate weight - highest in the middle of sequence, lower at edges
            # This helps create smoother transitions between sequences
            position = offset / (seq_length - 1)  # 0 to 1
            # Triangular weight function peaking at center of sequence
            weight = 1.0 - 2.0 * abs(position - 0.5)
            weight = max(0.1, weight)  # Ensure minimum weight of 0.1
            
            # Add weighted prediction to this frame
            frame_predictions[seq_start_idx + offset] += sequence_predictions[seq_idx] * weight
            frame_coverage[seq_start_idx + offset] += weight
    
    # For frames covered by multiple sequences, take the weighted average
    for i in range(len(frame_predictions)):
        if frame_coverage[i] > 0:
            frame_predictions[i] /= frame_coverage[i]
    
    # For frames not covered by any sequence (early frames), use nearest prediction
    if len(sequence_predictions) > 0:
        for i in range(len(frame_predictions)):
            if frame_coverage[i] == 0:
                # Find nearest covered frame
                covered_frames = np.where(frame_coverage > 0)[0]
                if len(covered_frames) > 0:
                    nearest_frame = covered_frames[np.argmin(np.abs(covered_frames - i))]
                    frame_predictions[i] = frame_predictions[nearest_frame]
                else:
                    frame_predictions[i] = sequence_predictions[0]
    
    # Apply additional temporal smoothing using a moving average
    window_size = 5
    frame_smoothed = np.copy(frame_predictions)
    
    for i in range(len(frame_predictions)):
        # Calculate window boundaries
        window_start = max(0, i - window_size // 2)
        window_end = min(len(frame_predictions), i + window_size // 2 + 1)
        
        # Calculate weighted moving average
        weights = np.exp(-0.5 * np.square(np.arange(window_start - i, window_end - i) / (window_size / 4)))
        frame_smoothed[i] = np.sum(frame_predictions[window_start:window_end] * weights) / np.sum(weights)
    
    # Convert to binary predictions
    frame_binary_predictions = [1 if pred >= config.THRESHOLD else 0 for pred in frame_smoothed]
    
    return frame_smoothed, frame_binary_predictions

def visualize_prediction_comparison(frame_files, predictions, binary_predictions, 
                                    temporal_predictions, temporal_binary_predictions,
                                    output_folder, num_samples=5):
    """Visualize a comparison of frame-based and temporal predictions"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get indices where the predictions differ
    diff_indices = [i for i in range(len(binary_predictions)) 
                   if binary_predictions[i] != temporal_binary_predictions[i]]
    
    if diff_indices:
        # Sample from differing predictions
        sample_indices = np.random.choice(diff_indices, min(num_samples, len(diff_indices)), replace=False)
    else:
        # If no differences, just sample randomly
        sample_indices = np.random.choice(len(binary_predictions), min(num_samples, len(binary_predictions)), replace=False)
    
    # Create visualization
    for idx in sample_indices:
        frame_file = frame_files[idx]
        
        # Original model predictions
        pred_score = predictions[idx]
        binary_pred = binary_predictions[idx]
        
        # Temporal model predictions
        temporal_score = temporal_predictions[idx]
        temporal_binary_pred = temporal_binary_predictions[idx]
        
        # Load image
        img = cv2.imread(frame_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original model plot
        ax1.imshow(img)
        status = "LICK" if binary_pred == 1 else "NO LICK"
        ax1.set_title(f"Frame-based: {status} (Score: {pred_score:.4f})")
        ax1.axis('off')
        
        # Temporal model plot
        ax2.imshow(img)
        status = "LICK" if temporal_binary_pred == 1 else "NO LICK"
        ax2.set_title(f"Temporal: {status} (Score: {temporal_score:.4f})")
        ax2.axis('off')
        
        # Add overall title
        plt.suptitle(f"Prediction Comparison - Frame {os.path.basename(frame_file)}")
        
        # Save figure
        output_file = os.path.join(output_folder, f"comparison_{os.path.basename(frame_file)}")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison visualizations saved to {output_folder}")

def plot_prediction_comparison_timeline(predictions, binary_predictions, 
                                       temporal_predictions, temporal_binary_predictions, 
                                       output_file):
    """Plot a comparison of frame-based and temporal predictions over time"""
    plt.figure(figsize=(15, 8))
    
    # Create two subplots vertically stacked
    grid = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # First subplot: Original model
    ax1 = plt.subplot(grid[0])
    ax1.plot(predictions, 'b-', alpha=0.6, label='Prediction Score')
    ax1.plot(binary_predictions, 'r-', alpha=0.5, label='Binary Prediction')
    ax1.axhline(y=config.THRESHOLD, color='g', linestyle='--', label=f'Threshold ({config.THRESHOLD})')
    ax1.set_title("Frame-based Model Predictions")
    ax1.set_ylabel("Prediction Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Second subplot: Temporal model
    ax2 = plt.subplot(grid[1])
    ax2.plot(temporal_predictions, 'b-', alpha=0.6, label='Prediction Score')
    ax2.plot(temporal_binary_predictions, 'r-', alpha=0.5, label='Binary Prediction')
    ax2.axhline(y=config.THRESHOLD, color='g', linestyle='--', label=f'Threshold ({config.THRESHOLD})')
    ax2.set_title("Temporal Model Predictions")
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Prediction Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison timeline plot saved to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict lick events in video frames using temporal model')
    parser.add_argument('--model', type=str, default=config.BEST_TEMPORAL_MODEL_PATH, 
                        help=f'Path to the trained temporal model checkpoint (default: {config.BEST_TEMPORAL_MODEL_PATH})')
    parser.add_argument('--original-model', type=str, default=config.BEST_MODEL_PATH,
                        help=f'Path to the trained original model checkpoint (default: {config.BEST_MODEL_PATH})')
    parser.add_argument('--frames', type=str, default=None, help='Path to the folder containing frames')
    parser.add_argument('--output', type=str, default=None, help='Output file to save predictions')
    parser.add_argument('--threshold', type=float, default=config.THRESHOLD, 
                        help=f'Threshold for binary classification (default: {config.THRESHOLD})')
    parser.add_argument('--animal', type=str, default=config.ANIMAL_ID,
                       help=f'Animal ID (default: {config.ANIMAL_ID})')
    parser.add_argument('--sequence-length', type=int, default=config.SEQUENCE_LENGTH,
                        help=f'Number of frames in each sequence (default: {config.SEQUENCE_LENGTH})')
    parser.add_argument('--sequence-stride', type=int, default=1,  # Always use stride=1
                        help=f'Step size between sequences (default: 1)')
    parser.add_argument('--compare', action='store_true', help='Compare with original frame-based model')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample predictions')
    parser.add_argument('--random-video', action='store_true', help='Use a random video if no frames path specified')
    args = parser.parse_args()
    
    # If no frames path specified and random-video flag is set, select a random video
    if args.frames is None and args.random_video:
        print("No frames path specified, selecting a random video...")
        # Search for videos with labels
        try:
            # Try to load dataset that has labels
            dataset_pkl = r"C:/Users/Nuo Lab/Desktop/lick_proj/swallow_lick_breath_tracking_dataset.pkl"
            with open(dataset_pkl, 'rb') as f:
                data = pickle.load(f)
            
            animal_ids = data["animal_id"]
            session_dates = [date.replace('-', '_') for date in data["session_date"]]
            side_tracking_all = data["side_tracking"]  # Contains frame-by-frame lick labels
            
            # Get random animal, session, and trial
            random_idx = np.random.randint(0, len(animal_ids))
            animal_id = animal_ids[random_idx]
            session_date = session_dates[random_idx]
            session_data = side_tracking_all[random_idx]
            video_trials = session_data[2]
            
            # Find a valid trial with labels
            if isinstance(video_trials, (list, np.ndarray)) and len(video_trials) > 0:
                trial_idx = np.random.randint(0, len(video_trials))
                
                # Try to find the corresponding video file
                video_paths = [
                    f'I:/side/{animal_id}/{session_date}',
                    f'I:/videos/side/{animal_id}/{session_date}',
                    f'C:/Users/Nuo Lab/Desktop/lick_proj/G/videos/side/{animal_id}/{session_date}',
                    f'I:/videos/{animal_id}/{session_date}'
                ]
                
                video_found = False
                for video_path in video_paths:
                    if os.path.exists(video_path):
                        # Get video files
                        video_files = [f for f in os.listdir(video_path) 
                                    if f.endswith('.mp4') and animal_id in f and 
                                    session_date in f and f"_{trial_idx}." in f]
                        
                        if video_files:
                            # Extract frames from the first video
                            video_file = video_files[0]
                            full_video_path = os.path.join(video_path, video_file)
                            print(f"Selected video: {full_video_path}")
                            
                            # Create temporary folder for frames
                            temp_frames_dir = os.path.join("temp_frames", os.path.basename(video_file).split('.')[0])
                            os.makedirs(temp_frames_dir, exist_ok=True)
                            
                            # Extract frames using OpenCV
                            cap = cv2.VideoCapture(full_video_path)
                            frame_count = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Save frame
                                frame_path = os.path.join(temp_frames_dir, f"frame_{frame_count:06d}.jpg")
                                cv2.imwrite(frame_path, frame)
                                frame_count += 1
                            
                            cap.release()
                            
                            if frame_count > 0:
                                print(f"Extracted {frame_count} frames to {temp_frames_dir}")
                                args.frames = temp_frames_dir
                                video_found = True
                                break
                    
                if not video_found:
                    print("Could not find any video file for selected animal/session/trial")
            else:
                print("No trials with labels found in the selected session")
        
        except Exception as e:
            print(f"Error selecting random video: {e}")
    
    # If still no frames path, exit
    if args.frames is None:
        print("Error: No frames path specified. Use --frames to specify a path or --random-video to select a random video.")
        return
    
    # Print configuration
    print(f"Using configuration:")
    print(f"  - Animal ID: {args.animal}")
    print(f"  - Model: {args.model}")
    print(f"  - Threshold: {args.threshold}")
    print(f"  - Sequence length: {args.sequence_length}")
    print(f"  - Sequence stride: 1")  # Always use stride 1
    print(f"  - Frames path: {args.frames}")
    
    # Load model
    model = load_trained_model(args.model)
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    
    if not frame_files:
        print(f"No frames found in {args.frames}")
        return
    
    print(f"Found {len(frame_files)} frames in {args.frames}")
    
    # Create sequences from frames
    sequences, sequence_to_frame_idx = create_frame_sequences(
        frame_files, 
        sequence_length=args.sequence_length,
        sequence_stride=1  # Always use stride 1
    )
    
    print(f"Created {len(sequences)} sequences")
    
    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Predict sequences
    sequence_predictions, sequence_binary_predictions = predict_sequences(
        model, 
        sequences, 
        transform, 
        threshold=args.threshold
    )
    
    # Map sequence predictions back to frames
    frame_predictions, frame_binary_predictions = map_sequence_predictions_to_frames(
        frame_files,
        sequence_to_frame_idx,
        sequence_predictions,
        sequence_binary_predictions
    )
    
    # Create output directory if needed
    if args.output:
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        results = {
            'frame_files': frame_files,
            'predictions': frame_predictions,
            'binary_predictions': frame_binary_predictions,
            'threshold': args.threshold,
            'sequence_length': args.sequence_length,
            'sequence_stride': args.sequence_stride
        }
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
        print(f"Predictions saved to {args.output}")
        
        # Plot prediction timeline
        timeline_plot = os.path.join(output_dir, "temporal_prediction_timeline.png")
        plt.figure(figsize=(15, 5))
        plt.plot(frame_predictions, 'b-', alpha=0.6, label='Prediction Score')
        plt.plot(frame_binary_predictions, 'r-', alpha=0.5, label='Binary Prediction')
        plt.axhline(y=args.threshold, color='g', linestyle='--', label=f'Threshold ({args.threshold})')
        plt.title("Temporal Model Lick Event Predictions Over Time")
        plt.xlabel("Frame Number")
        plt.ylabel("Prediction Score")
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.savefig(timeline_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Timeline plot saved to {timeline_plot}")
    
    # Compare with original model
    if args.compare:
        # Import original prediction function
        try:
            from predict_lick_events import load_trained_model as load_original_model
            from predict_lick_events import predict_frames
            
            original_model = load_original_model(args.original_model)
            print(f"Loaded original model for comparison")
            
            # Run prediction with original model
            original_predictions, original_binary_predictions = predict_frames(
                original_model,
                args.frames,
                output_file=None,
                threshold=args.threshold
            )
            
            # Compare predictions
            diff_count = sum(1 for i in range(len(frame_binary_predictions)) 
                            if frame_binary_predictions[i] != original_binary_predictions[i])
            diff_percentage = (diff_count / len(frame_binary_predictions)) * 100
            
            print(f"Comparison with original model:")
            print(f"  - Frames with different predictions: {diff_count} ({diff_percentage:.2f}%)")
            
            # Save comparison plot
            if args.output:
                comparison_plot = os.path.join(output_dir, "model_comparison_timeline.png")
                plot_prediction_comparison_timeline(
                    original_predictions, original_binary_predictions,
                    frame_predictions, frame_binary_predictions,
                    comparison_plot
                )
                
                # Visualize differences if requested
                if args.visualize:
                    comparison_viz_dir = os.path.join(output_dir, "comparison_visualizations")
                    visualize_prediction_comparison(
                        frame_files,
                        original_predictions, original_binary_predictions,
                        frame_predictions, frame_binary_predictions,
                        comparison_viz_dir
                    )
        except ImportError as e:
            print(f"Error importing original model: {e}")
            print("Comparison with original model skipped")

if __name__ == "__main__":
    main() 