# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:12:46 2025

@author: Nuo Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:24:23 2024

@author: Nuo Lab
"""
import os
import cv2
import pickle
import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import train_test_split
import config  # Import the centralized configuration

# Set a random seed for reproducibility
np.random.seed(42)

# Improved CUDA detection for OpenCV
def check_gpu():
    """Check if CUDA-enabled GPU is available through OpenCV"""
    # First check if CUDA module exists
    if not hasattr(cv2, 'cuda'):
        print("OpenCV was not built with CUDA support")
        return False
    
    # Then check if any compatible devices are detected
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            print(f"Found {cuda_devices} CUDA-enabled GPU(s) for OpenCV")
            return True
        else:
            print("OpenCV sees CUDA module but found 0 compatible devices")
            print("Using PyTorch for CUDA detection as fallback...")
            
            # Fallback to PyTorch CUDA detection
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"PyTorch found {torch.cuda.device_count()} CUDA device(s)")
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    # We'll proceed with CPU for OpenCV operations
                    return False
                else:
                    print("PyTorch also doesn't detect CUDA")
                    return False
            except ImportError:
                print("PyTorch not available for fallback check")
                return False
    except Exception as e:
        print(f"Error checking CUDA devices: {e}")
        return False
    
    return False

# Initialize GPU context if available
USE_GPU = check_gpu()
if USE_GPU:
    cv2.cuda.setDevice(0)  # Use the first GPU
else:
    print("Falling back to CPU for frame extraction")

# Path to the pickled dataset
dataset_pkl = r"C:/Users/Nuo Lab/Desktop/lick_proj/swallow_lick_breath_tracking_dataset.pkl"
view = config.VIEW  # Get view from config

# Constants
IMG_SIZE = (86, 130)  # Image size
THRESHOLD = config.THRESHOLD  # Use threshold from config 

# Load the pickle data with video and label information
with open(dataset_pkl, 'rb') as f:
    data = pickle.load(f)

# Extract data fields
animal_ids = data["animal_id"]
session_dates = [date.replace('-', '_') for date in data["session_date"]]
side_tracking_all = data["side_tracking"]  # Contains frame-by-frame lick labels

# Paths
video_root_path = f'I:/{view}'
output_root_path = f'I:/frames/{view}'

# Initialize list for missing videos
missing_videos = []

def extract_frames_gpu(video_path, output_folder, start_frame, end_frame):
    """
    Extract frames from a video using GPU acceleration when available.
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Directory to save the extracted frames
        start_frame (int): Starting frame number
        end_frame (int): Ending frame number
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust frame ranges
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, total_frames - 1)

    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    local_frame_counter = 0

    # Create GPU upload stream if using GPU
    if USE_GPU:
        stream = cv2.cuda.Stream()

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Unable to read frame {frame_idx} from {video_path}")
            break

        if USE_GPU:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame, stream)
            
            # Perform GPU-based processing if needed
            # e.g., resizing, color conversion, etc.
            # gpu_frame = cv2.cuda.resize(gpu_frame, (new_width, new_height))
            
            # Download processed frame back to CPU
            frame = gpu_frame.download(stream)

        frame_name = os.path.join(output_folder, f"frame_{local_frame_counter:06d}.jpg")
        cv2.imwrite(frame_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        frame_idx += 1
        local_frame_counter += 1

    cap.release()
    if USE_GPU:
        stream.waitForCompletion()
    
    return local_frame_counter  # Return the number of frames extracted

def process_video(video_file, animal_id, session_date, video_trials, animal_session_dir):
    """
    Process a single video file for multiprocessing.
    
    Returns:
        tuple: (output_folder, labels) if successful, (None, None) otherwise
    """
    try:
        # Extract trial index from the video filename
        trial_idx = int(video_file.split('_')[-1].split('.')[0])
        
        # Check if this trial index exists in the labels
        if trial_idx >= len(video_trials):
            print(f"  - Warning: Trial index {trial_idx} from {video_file} exceeds available labels ({len(video_trials)})")
            return None, None
        
        video_path = os.path.join(animal_session_dir, video_file)
        
        # Create output directory for frames
        output_folder = os.path.join(
            output_root_path, animal_id, session_date,
            f"trial_{trial_idx}"
        )
        
        # Get labels for this trial using the extracted trial index
        try:
            labels = np.array(video_trials[trial_idx])
            
            # Check total frames in the video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # If folder already exists and has frames, skip extraction
            if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
                print(f"  - Skipping already processed video: {video_file}")
            else:
                # Extract frames from 0 to total_frames-1
                print(f"  - Extracting frames from {video_file}")
                extract_frames_gpu(video_path, output_folder, 0, total_frames - 1)
            
            return output_folder, labels
            
        except Exception as e:
            print(f"  - Error processing {video_file}: {e}")
            return None, None
                
    except (ValueError, IndexError) as e:
        print(f"  - Cannot extract trial index from {video_file}: {e}")
        return None, None

def process_animal_data(animal_id, animal_indices):
    """
    Process data for a single animal with GPU acceleration and multiprocessing.
    
    Returns:
        video_paths_lick (list of str): Folders with extracted frames (lick dataset).
        labels_lick (list of np.ndarray): Corresponding lick labels for each video.
    """
    print(f"\nProcessing Animal: {animal_id}")
    
    video_paths_lick = []
    labels_lick = []
    
    for idx in animal_indices:
        session_date = session_dates[idx]
        session_data = side_tracking_all[idx]
        
        # Skip if there's not enough session data
        if len(session_data) < 3:
            print(f"  - Skipping session {session_date} due to insufficient data")
            continue
        
        # Get trial data which contains frame-by-frame lick labels
        video_trials = session_data[2]
        
        # Find all videos for this animal and session date
        animal_session_dir = os.path.join(video_root_path, animal_id, session_date)
        if not os.path.exists(animal_session_dir):
            print(f"  - Directory not found: {animal_session_dir}")
            continue
            
        # Get all video files in the directory
        video_files = [f for f in os.listdir(animal_session_dir) 
                      if f.endswith('.mp4') and animal_id in f and session_date in f]
        
        # Process videos in parallel
        process_func = partial(process_video, animal_id=animal_id, session_date=session_date, 
                             video_trials=video_trials, animal_session_dir=animal_session_dir)
        
        # Determine number of processes to use (use half of available cores to avoid overloading)
        num_processes = max(1, mp.cpu_count() // 2)
        print(f"  - Using {num_processes} processes for parallel video processing")
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_func, video_files)
        
        # Filter out None results and add valid ones to our lists
        for output_folder, labels in results:
            if output_folder is not None and labels is not None:
                video_paths_lick.append(output_folder)
                labels_lick.append(labels)
    
    return video_paths_lick, labels_lick

if __name__ == "__main__":
    # Main processing loop
    unique_animals = list(set(animal_ids))

    for animal_id in unique_animals:
        # Process only target animal
        target_animal = config.ANIMAL_ID  # Get target animal from config
        if animal_id != target_animal:
            print(f"Skipping animal {animal_id} (not the target animal {target_animal})")
            continue
            
        animal_indices = [i for i, aid in enumerate(animal_ids) if aid == animal_id]
        
        # Process the animal's data
        animal_video_paths_lick, animal_labels_lick = process_animal_data(animal_id, animal_indices)
        
        # If no videos found, skip
        if len(animal_video_paths_lick) == 0:
            print(f"  - No valid videos found for animal {animal_id}. Skipping train/test split.")
            continue
        
        # Train/Test Split for Lick data
        try:
            train_paths_lick, test_paths_lick, train_labels_lick_list, test_labels_lick_list = train_test_split(
                animal_video_paths_lick, 
                animal_labels_lick, 
                test_size=0.2,     # 80% train, 20% test
                random_state=42
            )
            
            train_labels_lick_dict = {
                path: labels for path, labels in zip(train_paths_lick, train_labels_lick_list)
            }
            test_labels_lick_dict = {
                path: labels for path, labels in zip(test_paths_lick, test_labels_lick_list)
            }
            
            # Save lick labels using helper functions from config
            os.makedirs(os.path.dirname(config.get_train_labels_path()), exist_ok=True)
            
            with open(config.get_train_labels_path(), 'wb') as f:
                pickle.dump(train_labels_lick_dict, f)
            print(f"  - Lick Train labels saved to {config.get_train_labels_path()}")
            
            with open(config.get_test_labels_path(), 'wb') as f:
                pickle.dump(test_labels_lick_dict, f)
            print(f"  - Lick Test labels saved to {config.get_test_labels_path()}")
            
        except ValueError as e:
            print(f"  - Error splitting lick data for animal {animal_id}: {e}")
            continue

    # Print missing videos summary
    if missing_videos:
        print("\nVideos with missing files:")
        for video in missing_videos:
            print(f"  - {video}")
    else:
        print("\nAll videos were found.")

    print("\nMultiprocessing-enabled preprocessing completed successfully.")
