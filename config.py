"""
Configuration settings for the lick detection pipeline.
This file centralizes all shared configuration settings used across 
the data processing, training, and evaluation scripts.
"""

# Animal and data settings
ANIMAL_ID = "DL005"  # Change this in one place to update all scripts
VIEW = "side"  # The camera view to use (side, front, etc.)

# Data paths
DATA_ROOT = "C:/Users/Nuo Lab/Desktop/lick/withframenew"
FRAME_ROOT = "I:/frames/side"  # For frame extraction and prediction

# Model settings
MODEL_DIR = "models/lick_classifier"
BEST_MODEL_PATH = f"{MODEL_DIR}/lick_classifier_best.pth"

# Classification settings
THRESHOLD = 0.5  # Threshold for binary classification

# Image processing settings
IMAGE_SIZE = (224, 224)  # Size to resize images to for the model

# Training settings
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
TENSORBOARD_LOG_DIR = "runs/lick_classifier"

# Temporal model settings
USE_TEMPORAL_MODEL = True  # Whether to use the temporal model
SEQUENCE_LENGTH = 30  # Increased to 30 as requested
LSTM_HIDDEN_SIZE = 64  # Reduced from 128 to save memory
LSTM_NUM_LAYERS = 1  # Using single layer to save memory
SEQUENCE_STRIDE = 15  # Increased stride to reduce total sequences
TEMPORAL_BATCH_SIZE = 2  # Drastically reduced batch size for long sequences
GRADIENT_ACCUMULATION_STEPS = 8  # Use gradient accumulation to simulate larger batches
TEMPORAL_MODEL_DIR = "models/temporal_lick_classifier"
BEST_TEMPORAL_MODEL_PATH = f"{TEMPORAL_MODEL_DIR}/temporal_lick_classifier_best.pth"
TEMPORAL_TENSORBOARD_LOG_DIR = "runs/temporal_lick_classifier"

# Paths for saving data
def get_train_labels_path():
    """Get the path for training labels"""
    return f"{DATA_ROOT}/datatags_{VIEW}/train_labels_lick_{ANIMAL_ID}.pkl"

def get_test_labels_path():
    """Get the path for test labels"""
    return f"{DATA_ROOT}/datatags_{VIEW}/test_labels_lick_{ANIMAL_ID}.pkl"

def get_frames_dir(animal_id=None):
    """Get the directory containing frames for the specified animal"""
    if animal_id is None:
        animal_id = ANIMAL_ID
    return f"{FRAME_ROOT}/{animal_id}"

# Evaluation settings
EVAL_RESULTS_DIR = "evaluation_results" 