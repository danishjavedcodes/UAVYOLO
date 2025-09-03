import os
import torch
import csv
from torch.utils.data import DataLoader
from models.faster_rcnn import get_hmaytsf_baseline_model as get_model
from datasets.custom_dataset import CustomDataset
from utils.transforms import get_transform
from utils.visualization import plot_training_history
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_model_metrics(model, data_loader, device, annotation_file):
    """
    Evaluate model and return comprehensive metrics including loss, precision, recall, mAP50, and mAP50-95
    """
    coco_gt = COCO(annotation_file)
    results = []
    total_loss = 0.0
    num_batches = 0
    
    model.eval()
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Calculate validation loss
            model.train()  # Temporarily set to train mode to get loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1
            
            # Set back to eval mode for inference
            model.eval()
            outputs = model(images)
            
            # Process predictions for COCO evaluation
            for i, output in enumerate(outputs):
                try:
                    if isinstance(targets[i], dict):
                        image_id = targets[i]["image_id"].item()
                    else:
                        image_id = i
                except (KeyError, TypeError):
                    image_id = i
                
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": float(score)
                    })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Handle case where no predictions were made
    if not results:
        return {
            "loss": avg_loss,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0
        }
    
    # COCO evaluation
    try:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        stats = coco_eval.stats
        
        return {
            "loss": avg_loss,
            "precision": float(stats[1]),    # AP@0.5 (used as precision proxy)
            "recall": float(stats[8]),       # AR@0.5:0.95 (average recall)
            "mAP50": float(stats[1]),        # mAP@0.5
            "mAP50_95": float(stats[0])      # mAP@0.5:0.95
        }
    except Exception as e:
        print(f"COCO evaluation failed: {e}")
        return {
            "loss": avg_loss,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0
        }


def train_model(model, train_loader, eval_loader, optimizer, device, num_epochs, results_dir, annotation_file):
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize CSV file for metrics logging
    csv_path = os.path.join(results_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'mAP50', 'mAP50_95']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Initialize tracking for best model
    best_map50 = 0.0
    best_model_path = os.path.join(results_dir, 'best_model.pth')

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}/{num_epochs}')

        for images, targets in train_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            train_bar.set_postfix(loss=f'{losses.item():.4f}')

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}')

        # Evaluation phase
        print("Running evaluation...")
        eval_metrics = evaluate_model_metrics(model, eval_loader, device, annotation_file)
        
        # Print evaluation results
        print(f'Validation Loss: {eval_metrics["loss"]:.4f}')
        print(f'Precision: {eval_metrics["precision"]:.4f}')
        print(f'Recall: {eval_metrics["recall"]:.4f}')
        print(f'mAP50: {eval_metrics["mAP50"]:.4f}')
        print(f'mAP50-95: {eval_metrics["mAP50_95"]:.4f}')
        
        # Save metrics to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'mAP50', 'mAP50_95']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': eval_metrics['loss'],
                'precision': eval_metrics['precision'],
                'recall': eval_metrics['recall'],
                'mAP50': eval_metrics['mAP50'],
                'mAP50_95': eval_metrics['mAP50_95']
            })
        
        # Model checkpointing logic
        
        # Save best model (based on mAP50)
        if eval_metrics['mAP50'] > best_map50:
            best_map50 = eval_metrics['mAP50']
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved! mAP50: {best_map50:.4f} at {best_model_path}')
        
        # Save model every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(results_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at: {checkpoint_path}')
        
        # Save last epoch model
        if epoch == num_epochs:
            last_epoch_path = os.path.join(results_dir, 'last_epoch_model.pth')
            torch.save(model.state_dict(), last_epoch_path)
            print(f'Last epoch model saved at: {last_epoch_path}')
        
        print("-" * 60)

