#!/usr/bin/env python3
"""
Pothole Detection Model Training Script using YOLOv8
This script trains a YOLOv8 model for pothole detection using GPU acceleration.
"""

import os
import sys
from ultralytics import YOLO
import torch

def check_gpu():
    """Check if GPU is available and print device information."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU available! Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("❌ No GPU available. Training will use CPU (much slower)")
        return False

def train_model():
    """Train the pothole detection model."""
    
    # Check GPU availability
    use_gpu = check_gpu()
    
    # Load the model
    print("📦 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model
    
    # Training configuration
    training_config = {
        'data': 'dataset.yaml',           # Path to dataset config
        'epochs': 100,                    # Number of epochs
        'imgsz': 640,                     # Image size
        'batch': 16,                      # Batch size (adjust based on GPU memory)
        'device': 0 if use_gpu else 'cpu', # Use GPU if available
        'workers': 8,                     # Number of worker threads
        'patience': 50,                   # Early stopping patience
        'save': True,                     # Save checkpoints
        'save_period': 10,                # Save every 10 epochs
        'cache': False,                   # Cache images for faster training
        'amp': True,                      # Automatic mixed precision
        'cos_lr': True,                   # Cosine learning rate scheduler
        'close_mosaic': 10,               # Close mosaic augmentation in last 10 epochs
        'lr0': 0.01,                      # Initial learning rate
        'lrf': 0.01,                      # Final learning rate
        'momentum': 0.937,                # SGD momentum
        'weight_decay': 0.0005,           # Weight decay
        'warmup_epochs': 3.0,             # Warmup epochs
        'warmup_momentum': 0.8,           # Warmup momentum
        'warmup_bias_lr': 0.1,            # Warmup bias learning rate
        'box': 7.5,                       # Box loss gain
        'cls': 0.5,                       # Class loss gain
        'dfl': 1.5,                       # DFL loss gain
        'hsv_h': 0.015,                   # HSV-Hue augmentation
        'hsv_s': 0.7,                     # HSV-Saturation augmentation
        'hsv_v': 0.4,                     # HSV-Value augmentation
        'degrees': 0.0,                   # Image rotation
        'translate': 0.1,                 # Image translation
        'scale': 0.5,                     # Image scaling
        'shear': 0.0,                     # Image shear
        'perspective': 0.0,               # Image perspective
        'flipud': 0.0,                    # Image flip up-down
        'fliplr': 0.5,                    # Image flip left-right
        'mosaic': 1.0,                    # Mosaic augmentation
        'mixup': 0.0,                     # Mixup augmentation
        'copy_paste': 0.0,                # Copy-paste augmentation
        'auto_augment': 'randaugment',    # Auto augmentation
        'erasing': 0.4,                   # Random erasing
        'verbose': True,                  # Verbose output
        'project': 'runs',                # Project name
        'name': 'pothole_detection',      # Experiment name
        'exist_ok': True,                 # Overwrite existing experiment
    }
    
    print("🚀 Starting training...")
    print(f"📊 Training configuration:")
    print(f"   - Model: YOLOv8n")
    print(f"   - Epochs: {training_config['epochs']}")
    print(f"   - Batch size: {training_config['batch']}")
    print(f"   - Image size: {training_config['imgsz']}")
    print(f"   - Device: {'GPU' if use_gpu else 'CPU'}")
    print(f"   - Dataset: {training_config['data']}")
    
    # Start training
    try:
        results = model.train(**training_config)
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\n📈 Final Training Metrics:")
            for metric, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        return None

def validate_model():
    """Validate the trained model."""
    print("\n🔍 Validating model...")
    
    # Load the best model
    best_model_path = 'runs/pothole_detection/weights/best.pt'
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)
        
        # Run validation
        results = model.val()
        
        print("✅ Validation completed!")
        print(f"📊 Validation metrics:")
        if hasattr(results, 'results_dict'):
            for metric, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
    else:
        print("❌ Best model not found. Please check if training completed successfully.")

if __name__ == "__main__":
    print("🔧 Pothole Detection Model Training")
    print("=" * 50)
    
    # Check if dataset.yaml exists
    if not os.path.exists('dataset.yaml'):
        print("❌ dataset.yaml not found! Please ensure it exists in the current directory.")
        sys.exit(1)
    
    # Check if dataset directories exist
    import yaml
    with open('dataset.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    train_path = dataset_config.get('train')
    val_path = dataset_config.get('val')
    
    if not os.path.exists(train_path):
        print(f"❌ Training data path not found: {train_path}")
        sys.exit(1)
    
    if not os.path.exists(val_path):
        print(f"❌ Validation data path not found: {val_path}")
        sys.exit(1)
    
    print(f"✅ Dataset paths verified:")
    print(f"   - Training: {train_path}")
    print(f"   - Validation: {val_path}")
    
    # Train the model
    results = train_model()
    
    if results:
        # Validate the model
        validate_model()
        
        print("\n🎉 Training pipeline completed!")
        print("📁 Check the 'runs/pothole_detection' directory for results and model weights.")
    else:
        print("\n❌ Training failed. Please check the error messages above.") 