import torch
import torchvision.models as models
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_model():
    """Create the model architecture used during training"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

def create_temporal_model(sequence_length=10, hidden_size=128, num_layers=2):
    """Create a model that incorporates temporal information using LSTM layers
    
    Args:
        sequence_length: Number of frames in each sequence
        hidden_size: Size of the LSTM hidden state
        num_layers: Number of LSTM layers
        
    Returns:
        A model that processes sequences of frames
    """
    # Base CNN model for feature extraction (ResNet18 with pretrained weights)
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Memory optimization: Create a feature extractor that stops at layer3
    # This reduces the feature dimension from 512 to 256
    feature_extractor = nn.Sequential(
        base_model.conv1,
        base_model.bn1,
        base_model.relu,
        base_model.maxpool,
        base_model.layer1,
        base_model.layer2,
        base_model.layer3,  # Stop at layer3 (256 channels)
        nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
    )
    
    # Get the reduced feature dimension
    cnn_output_size = 256  # Reduced from 512 to 256
    
    # Create the LSTM-based temporal model
    class TemporalModel(nn.Module):
        def __init__(self):
            super(TemporalModel, self).__init__()
            # CNN feature extractor (now using pretrained weights)
            self.feature_extractor = feature_extractor
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=cnn_output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            
            # Final prediction layer
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),  # Reduced from 64 to 32
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1)  # Remove sigmoid since we're using BCEWithLogitsLoss
            )
            
            # Temporal smoothing layer for inference
            self.smoothing = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                padding=2,
                bias=False
            )
            # Initialize with gaussian-like weights for smoothing
            nn.init.normal_(self.smoothing.weight, mean=0.2, std=0.1)
            
        def forward(self, x):
            # Input shape: [batch_size, sequence_length, channels, height, width]
            batch_size, seq_len, c, h, w = x.shape
            
            # Process in chunks to save memory for very long sequences
            chunk_size = 10  # Process 10 frames at a time
            features_list = []
            
            for i in range(0, seq_len, chunk_size):
                # Get current chunk
                end_idx = min(i + chunk_size, seq_len)
                chunk = x[:, i:end_idx, :, :, :]
                chunk_size_actual = end_idx - i
                
                # Reshape for feature extraction
                chunk_reshaped = chunk.contiguous().view(batch_size * chunk_size_actual, c, h, w)
                
                # Extract features (memory efficient) - removed mixed precision to avoid float16 issues
                chunk_features = self.feature_extractor(chunk_reshaped)
                
                # Reshape to [batch_size, chunk_size, cnn_output_size]
                chunk_features = chunk_features.view(batch_size, chunk_size_actual, cnn_output_size)
                features_list.append(chunk_features)
            
            # Concatenate all chunks
            features = torch.cat(features_list, dim=1)
            
            # Process with LSTM
            lstm_out, _ = self.lstm(features)
            
            # Use the final output for prediction
            final_output = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_size]
            
            # Final prediction - will output shape [batch_size, 1]
            output = self.fc(final_output)
            
            # Keep the batch dimension even if batch size is 1
            if output.dim() == 1 and output.size(0) == 1:
                output = output.unsqueeze(0)  # Add batch dimension back
                
            # Ensure the output is properly shaped - [batch_size] for BCEWithLogitsLoss
            output = output.squeeze(-1)  # Remove the last dimension if needed
            
            # Apply temporal smoothing during inference
            if not self.training and output.dim() > 0:  # Check if not scalar
                # Make sure we're working with batch dim for smoothing
                # Reshape for 1D convolution
                output_reshaped = output.view(1, 1, -1)
                output_smoothed = self.smoothing(output_reshaped)
                output = output_smoothed.view(-1)  # Squeeze to [batch_size]
                
                # Handle single-sample case
                if output.dim() == 0:
                    output = output.unsqueeze(0)  # Ensure output is [1]
            
            return output
    
    return TemporalModel()

def visualize_model_architecture():
    # Create the model
    model = create_model()
    
    # Create a dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    y = model(x)
    
    # Generate computational graph
    dot = make_dot(y, params=dict(list(model.named_parameters())))
    
    # Save the graph
    dot.format = 'png'
    dot.render('model_architecture')
    
    print("Model architecture visualization saved as model_architecture.png")

def visualize_temporal_model_architecture():
    # Create the temporal model
    model = create_temporal_model(sequence_length=10)
    
    # Create a dummy input (batch_size, sequence_length, channels, height, width)
    x = torch.randn(1, 10, 3, 224, 224)
    
    # Forward pass
    y = model(x)
    
    # Generate computational graph
    dot = make_dot(y, params=dict(list(model.named_parameters())))
    
    # Save the graph
    dot.format = 'png'
    dot.render('temporal_model_architecture')
    
    print("Temporal model architecture visualization saved as temporal_model_architecture.png")

def create_model_diagram():
    # Create a visual representation of the ResNet-18 architecture
    # This is a simplified version for the report
    
    # Create a new image
    width, height = 1200, 600
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    font_path = None
    try:
        # Try to find a system font
        font_path = "arial.ttf"  # Common font on Windows
        font = ImageFont.truetype(font_path, 20)
        title_font = ImageFont.truetype(font_path, 24)
    except:
        # Fallback to default
        font = ImageFont.load_default()
        title_font = font
    
    # Draw title
    draw.text((width//2 - 200, 20), "ResNet-18 for Lick Detection", fill=(0, 0, 0), font=title_font)
    
    # Define the main blocks of ResNet
    blocks = [
        {"name": "Input", "size": "224×224×3", "position": (100, 100), "width": 120, "height": 60, "color": (173, 216, 230)},
        {"name": "Conv1", "size": "112×112×64", "position": (300, 100), "width": 120, "height": 60, "color": (144, 238, 144)},
        {"name": "MaxPool", "size": "56×56×64", "position": (500, 100), "width": 120, "height": 60, "color": (144, 238, 144)},
        {"name": "ResBlock 1", "size": "56×56×64", "position": (700, 100), "width": 120, "height": 60, "color": (255, 192, 203)},
        {"name": "ResBlock 2", "size": "28×28×128", "position": (900, 100), "width": 120, "height": 60, "color": (255, 192, 203)},
        
        {"name": "ResBlock 3", "size": "14×14×256", "position": (100, 250), "width": 120, "height": 60, "color": (255, 192, 203)},
        {"name": "ResBlock 4", "size": "7×7×512", "position": (300, 250), "width": 120, "height": 60, "color": (255, 192, 203)},
        {"name": "AvgPool", "size": "1×1×512", "position": (500, 250), "width": 120, "height": 60, "color": (144, 238, 144)},
        {"name": "FC", "size": "512→1", "position": (700, 250), "width": 120, "height": 60, "color": (255, 165, 0)},
        {"name": "Sigmoid", "size": "Score: 0-1", "position": (900, 250), "width": 120, "height": 60, "color": (255, 165, 0)},
    ]
    
    # Draw the blocks and connections
    for i, block in enumerate(blocks):
        # Draw block
        x, y = block["position"]
        w, h = block["width"], block["height"]
        color = block["color"]
        
        draw.rectangle([x, y, x+w, y+h], fill=color, outline=(0, 0, 0))
        
        # Draw text
        draw.text((x+10, y+10), block["name"], fill=(0, 0, 0), font=font)
        draw.text((x+10, y+30), block["size"], fill=(0, 0, 0), font=font)
        
        # Draw arrow to next block if not the last one
        if i < len(blocks) - 1:
            next_block = blocks[i+1]
            next_x, next_y = next_block["position"]
            
            # Check if we're moving to the next row
            if next_y > y:
                # Draw arrow from current block to the right edge
                draw.line([x+w, y+h//2, width-50, y+h//2], fill=(0, 0, 0), width=2)
                # Draw arrow from right edge down to next row
                draw.line([width-50, y+h//2, width-50, next_y+h//2], fill=(0, 0, 0), width=2)
                # Draw arrow from left edge to next block
                draw.line([50, next_y+h//2, next_x, next_y+h//2], fill=(0, 0, 0), width=2)
                # Arrow at end
                draw.polygon([(next_x-10, next_y+h//2-5), (next_x, next_y+h//2), (next_x-10, next_y+h//2+5)], fill=(0, 0, 0))
            else:
                # Direct arrow to next block
                draw.line([x+w, y+h//2, next_x, next_y+h//2], fill=(0, 0, 0), width=2)
                # Arrow at end
                draw.polygon([(next_x-10, next_y+h//2-5), (next_x, next_y+h//2), (next_x-10, next_y+h//2+5)], fill=(0, 0, 0))
    
    # Add explanation text
    explanation = [
        "Model: ResNet-18 (He et al., 2016)",
        "Total Parameters: ~11.7M",
        "Input: RGB frames (224×224)",
        "Output: Binary lick classification",
        "Threshold: 0.5"
    ]
    
    for i, text in enumerate(explanation):
        draw.text((width//2 - 150, 350 + i*30), text, fill=(0, 0, 0), font=font)
    
    # Save the image
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/model_diagram.png'
    img.save(output_path)
    print(f"Model diagram saved to {output_path}")

def create_temporal_model_diagram():
    # Create a visual representation of the Temporal model architecture
    # This is a simplified version for the report
    
    # Create a new image
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    font_path = None
    try:
        # Try to find a system font
        font_path = "arial.ttf"  # Common font on Windows
        font = ImageFont.truetype(font_path, 20)
        title_font = ImageFont.truetype(font_path, 24)
    except:
        # Fallback to default
        font = ImageFont.load_default()
        title_font = font
    
    # Draw title
    draw.text((width//2 - 260, 20), "Temporal LSTM Model for Lick Detection", fill=(0, 0, 0), font=title_font)
    
    # Define the main blocks
    blocks = [
        {"name": "Input Sequence", "size": "Frames: 1 to N", "position": (100, 100), "width": 150, "height": 60, "color": (173, 216, 230)},
        
        {"name": "ResNet18", "size": "Feature Extractor", "position": (350, 100), "width": 150, "height": 60, "color": (144, 238, 144)},
        {"name": "Features", "size": "512 features", "position": (600, 100), "width": 150, "height": 60, "color": (144, 238, 144)},
        
        {"name": "LSTM Layers", "size": "Temporal Processing", "position": (350, 250), "width": 150, "height": 60, "color": (255, 192, 203)},
        {"name": "FC Layers", "size": "Classification", "position": (600, 250), "width": 150, "height": 60, "color": (255, 165, 0)},
        {"name": "Output", "size": "Lick Probability", "position": (850, 250), "width": 150, "height": 60, "color": (255, 165, 0)},
    ]
    
    # Draw the sequence frames
    frame_positions = [(100, 200), (150, 200), (200, 200), (250, 200)]
    for x, y in frame_positions:
        # Draw frame
        draw.rectangle([x, y, x+40, y+40], fill=(173, 216, 230), outline=(0, 0, 0))
    
    # Draw ellipsis to indicate sequence
    draw.text((300, 210), "...", fill=(0, 0, 0), font=font)
    
    # Draw the blocks and connections
    for i, block in enumerate(blocks):
        # Draw block
        x, y = block["position"]
        w, h = block["width"], block["height"]
        color = block["color"]
        
        draw.rectangle([x, y, x+w, y+h], fill=color, outline=(0, 0, 0))
        
        # Draw text
        draw.text((x+10, y+10), block["name"], fill=(0, 0, 0), font=font)
        draw.text((x+10, y+30), block["size"], fill=(0, 0, 0), font=font)
    
    # Draw connection from sequence to ResNet
    draw.line([250, 190, 350, 130], fill=(0, 0, 0), width=2)
    draw.polygon([(340, 130), (350, 130), (345, 120)], fill=(0, 0, 0))
    
    # Draw connection from ResNet to Features
    draw.line([500, 130, 600, 130], fill=(0, 0, 0), width=2)
    draw.polygon([(590, 125), (600, 130), (590, 135)], fill=(0, 0, 0))
    
    # Draw connection from Features to LSTM
    draw.line([675, 160, 675, 200, 425, 200, 425, 250], fill=(0, 0, 0), width=2)
    draw.polygon([(420, 240), (425, 250), (430, 240)], fill=(0, 0, 0))
    
    # Draw connection from LSTM to FC
    draw.line([500, 280, 600, 280], fill=(0, 0, 0), width=2)
    draw.polygon([(590, 275), (600, 280), (590, 285)], fill=(0, 0, 0))
    
    # Draw connection from FC to Output
    draw.line([750, 280, 850, 280], fill=(0, 0, 0), width=2)
    draw.polygon([(840, 275), (850, 280), (840, 285)], fill=(0, 0, 0))
    
    # Add explanation text
    explanation = [
        "Model: ResNet-18 + LSTM",
        "Input: Sequence of frames",
        "ResNet: Feature extraction (frozen weights)",
        "LSTM: Temporal pattern recognition",
        "Output: Binary lick classification",
        "Sequence processing: Frame ordering matters"
    ]
    
    for i, text in enumerate(explanation):
        draw.text((width//2 - 200, 400 + i*30), text, fill=(0, 0, 0), font=font)
    
    # Save the image
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/temporal_model_diagram.png'
    img.save(output_path)
    print(f"Temporal model diagram saved to {output_path}")

if __name__ == "__main__":
    try:
        # Try to create visualization with torchviz
        visualize_model_architecture()
        visualize_temporal_model_architecture()
    except ImportError:
        print("torchviz not available, creating simplified diagram instead")
    
    # Create a simplified diagram that doesn't require additional libraries
    create_model_diagram()
    create_temporal_model_diagram() 