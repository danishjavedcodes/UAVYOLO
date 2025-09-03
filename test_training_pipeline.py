#!/usr/bin/env python3
"""
Test script to verify the updated training pipeline with metrics logging and checkpointing
"""

import os
import torch
import pandas as pd
from train import evaluate_model_metrics, train_model
from models.faster_rcnn import get_hmaytsf_baseline_model as get_model
from datasets.custom_dataset import CustomDataset
from utils.transforms import get_transform
from torch.utils.data import DataLoader
import torch.optim as optim

def test_training_pipeline():
    """Test the training pipeline with a small number of epochs"""
    
    # Test configuration
    ROOT_DIR_TRAIN = "./Aerial-Vehicles-1/train"
    ANNOTATION_FILE_TRAIN = "./Aerial-Vehicles-1/train/_annotations.coco.json"
    ROOT_DIR_EVAL = "./Aerial-Vehicles-1/valid"
    ANNOTATION_FILE_EVAL = "./Aerial-Vehicles-1/valid/_annotations.coco.json"
    RESULTS_DIR = "test_results"
    
    NUM_CLASSES = 5
    BATCH_SIZE = 2  # Small batch size for testing
    NUM_EPOCHS = 2  # Only 2 epochs for testing
    LEARNING_RATE = 0.005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"Testing training pipeline on {DEVICE}")
    
    # Check if data files exist
    if not os.path.exists(ANNOTATION_FILE_TRAIN):
        print(f"Warning: Training annotation file not found: {ANNOTATION_FILE_TRAIN}")
        print("Please ensure the dataset is available for testing")
        return False
        
    if not os.path.exists(ANNOTATION_FILE_EVAL):
        print(f"Warning: Evaluation annotation file not found: {ANNOTATION_FILE_EVAL}")
        print("Please ensure the dataset is available for testing")
        return False
    
    try:
        # Create datasets and loaders
        train_dataset = CustomDataset(ROOT_DIR_TRAIN, ANNOTATION_FILE_TRAIN, transform=get_transform(train=True))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        
        eval_dataset = CustomDataset(ROOT_DIR_EVAL, ANNOTATION_FILE_EVAL, transform=get_transform(train=False))
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        # Initialize model and optimizer
        model = get_model(NUM_CLASSES).to(DEVICE)
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
        
        print("Model and optimizer initialized successfully")
        
        # Test evaluation function
        print("Testing evaluation function...")
        eval_metrics = evaluate_model_metrics(model, eval_loader, DEVICE, ANNOTATION_FILE_EVAL)
        print("Evaluation metrics:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Test training pipeline
        print("Testing training pipeline...")
        train_model(model, train_loader, eval_loader, optimizer, DEVICE, NUM_EPOCHS, RESULTS_DIR, ANNOTATION_FILE_EVAL)
        
        # Verify outputs
        print("Verifying outputs...")
        
        # Check CSV file
        csv_path = os.path.join(RESULTS_DIR, 'training_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"CSV file created successfully with {len(df)} rows")
            print("CSV columns:", df.columns.tolist())
            print("Sample data:")
            print(df.head())
        else:
            print("ERROR: CSV file not created")
            return False
        
        # Check model files
        expected_files = [
            'best_model.pth',
            'last_epoch_model.pth'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(RESULTS_DIR, filename)
            if os.path.exists(filepath):
                print(f"✓ {filename} created successfully")
            else:
                print(f"✗ {filename} not found")
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_pipeline()
    if success:
        print("\n✅ All tests passed! The training pipeline is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
