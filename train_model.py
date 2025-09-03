# import os
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from PIL import Image
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# # Import custom modules
# from models.faster_rcnn import get_model
# from datasets.custom_dataset import CustomDataset
# from utils.transforms import get_transform
# from train import train_model

# # def evaluate_model(model, data_loader, device):
# #     coco_gt = COCO(ANNOTATION_FILE_EVAL)  # Ground truth annotations
# #     results = []

# #     with torch.no_grad():
# #         for images, targets in data_loader:
# #             images = list(image.to(device) for image in images)
# #             outputs = model(images)

# #             for i, output in enumerate(outputs):
# #                 image_id = targets[i]["image_id"].item()
# #                 boxes = output["boxes"].cpu().numpy()
# #                 scores = output["scores"].cpu().numpy()
# #                 labels = output["labels"].cpu().numpy()

# #                 for box, score, label in zip(boxes, scores, labels):
# #                     results.append({
# #                         "image_id": image_id,
# #                         "category_id": label,
# #                         "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # COCO format: [x, y, w, h]
# #                         "score": score
# #                     })

# #     # Save predictions to COCO format
# #     coco_dt = coco_gt.loadRes(results)

# #     # Evaluate using COCO API
# #     coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
# #     coco_eval.evaluate()
# #     coco_eval.accumulate()
# #     coco_eval.summarize()

# #     return coco_eval.stats  # Returns mAP and other metrics


# # if __name__ == "__main__":
# #     # Paths
# #     ROOT_DIR_TRAIN = "/home/danish/Desktop/Danish's/YOLO_NR/data/data/train"
# #     ANNOTATION_FILE_TRAIN = "/home/danish/Desktop/Danish's/YOLO_NR/data/data/train/_annotations.coco.json"
# #     ROOT_DIR_EVAL = "/home/danish/Desktop/Danish's/YOLO_NR/data/data/valid"
# #     ANNOTATION_FILE_EVAL = "/home/danish/Desktop/Danish's/YOLO_NR/data/data/_annotations.coco.json"
# #     MODEL_PATH = "/home/danish/Desktop/Danish's/YOLO_NR/object_detection_model_sar_data.pth"
# #     RESULTS_DIR = "results"

# #     # Hyperparameters
# #     NUM_CLASSES = 3
# #     BATCH_SIZE = 16
# #     NUM_EPOCHS = 150
# #     LEARNING_RATE = 0.005
# #     DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# #     # Create datasets and dataloaders for training
# #     train_dataset = CustomDataset(ROOT_DIR_TRAIN, ANNOTATION_FILE_TRAIN, transform=get_transform(train=True))
# #     train_loader = DataLoader(
# #         train_dataset,
# #         batch_size=BATCH_SIZE,
# #         shuffle=True,
# #         collate_fn=lambda x: tuple(zip(*x))
# #     )

# #     # Initialize model, optimizer, and move to device
# #     model = get_model(NUM_CLASSES)
# #     model.to(DEVICE)
# #     print("Model initialized successfully.")

# #     params = [p for p in model.parameters() if p.requires_grad]
# #     optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

# #     # Train the model
# #     print("Starting training...")
# #     train_model(model, train_loader, optimizer, DEVICE, num_epochs=NUM_EPOCHS, results_dir=RESULTS_DIR)

# #     # Save the trained model
# #     torch.save(model.state_dict(), MODEL_PATH)
# #     print(f"Model saved successfully at {MODEL_PATH}")


# import os
# import torch
# from tqdm import tqdm

# class EarlyStopping:
#     def __init__(self, patience=7, min_delta=0.001):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.best_loss = float('inf')
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter}/{self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True

# def train_model(model, data_loader, optimizer, device, num_epochs, results_dir, checkpoint_interval=10):
#     os.makedirs(results_dir, exist_ok=True)
#     early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         total_loss = 0

#         progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}/{num_epochs}')

#         for images, targets in progress_bar:
#             images = [img.to(device) for img in images]
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#             optimizer.zero_grad()
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#             losses.backward()
#             optimizer.step()

#             total_loss += losses.item()
#             progress_bar.set_postfix(loss=losses.item())

#         avg_loss = total_loss / len(data_loader)
#         print(f'Epoch {epoch}/{num_epochs} completed. Average Loss: {avg_loss:.4f}')

#         # Checkpoint saving every 10 epochs
#         if epoch % checkpoint_interval == 0 or epoch == num_epochs:
#             checkpoint_path = os.path.join(results_dir, f'model_epoch_{epoch}.pth')
#             torch.save(model.state_dict(), checkpoint_path)
#             print(f'Checkpoint saved at epoch {epoch} to {checkpoint_path}')

#         # Early stopping
#         early_stopping(avg_loss)
#         if early_stopping.early_stop:
#             print("Early stopping triggered.")
#             break

#     return model


# if __name__ == "__main__":
#     # Paths
#     ROOT_DIR_TRAIN = "/home/danish/Desktop/Danish's/YOLO_NR/data/data/train"
#     ANNOTATION_FILE_TRAIN = "/home/danish/Desktop/Danish's/YOLO_NR/data/data/train/_annotations.coco.json"
#     MODEL_PATH = "/home/danish/Desktop/Danish's/YOLO_NR/object_detection_model_sar_data.pth"
#     RESULTS_DIR = "results"

#     # Hyperparameters
#     NUM_CLASSES = 3
#     BATCH_SIZE = 16
#     NUM_EPOCHS = 150
#     LEARNING_RATE = 0.005
#     DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     print(f'Device: {DEVICE}')

#     # Dataset & DataLoader
#     train_dataset = CustomDataset(ROOT_DIR_TRAIN, ANNOTATION_FILE_TRAIN, transform=get_transform(train=True))
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         collate_fn=lambda x: tuple(zip(*x))
#     )

#     # Initialize Model and Optimizer
#     model = get_model(NUM_CLASSES)
#     model.to(DEVICE)
#     optimizer = optim.SGD(
#         [p for p in model.parameters() if p.requires_grad],
#         lr=LEARNING_RATE,
#         momentum=0.9,
#         weight_decay=0.0005
#     )

#     # Train Model
#     print("Starting training...")
#     trained_model = train_model(
#         model, 
#         train_loader, 
#         optimizer, 
#         DEVICE, 
#         num_epochs=NUM_EPOCHS, 
#         results_dir=RESULTS_DIR,
#         checkpoint_interval=10  # Save model every 10 epochs
#     )

#     # Save final model
#     torch.save(trained_model.state_dict(), MODEL_PATH)
#     print(f"Final model saved successfully at {MODEL_PATH}")


import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from datasets.custom_dataset import CustomDataset

# Import custom modules
from models.faster_rcnn import get_model
from datasets.custom_dataset import CustomDataset
from utils.transforms import get_transform
from train import train_model


if __name__ == "__main__":
    # Paths
    ROOT_DIR_TRAIN = "/home/danish/Desktop/Danish's/YOLO_NR/data/SAR/train"
    ANNOTATION_FILE_TRAIN = "/home/danish/Desktop/Danish's/YOLO_NR/data/SAR/train/_annotations.coco.json"
    ROOT_DIR_EVAL = "/home/danish/Desktop/Danish's/YOLO_NR/data/SAR/valid"
    ANNOTATION_FILE_EVAL = "/home/danish/Desktop/Danish's/YOLO_NR/data/SAR/valid/_annotations.coco.json"
    MODEL_PATH = "/home/danish/Desktop/Danish's/YOLO_NR/object_detection_model_SAR_data.pth"
    RESULTS_DIR = "results"

    # Hyperparameters
    NUM_CLASSES = 3
    BATCH_SIZE = 16
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {DEVICE}')

    # Training dataset and loader
    train_dataset = CustomDataset(ROOT_DIR_TRAIN, ANNOTATION_FILE_TRAIN, transform=get_transform(train=True))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Evaluation dataset and loader
    eval_dataset = CustomDataset(ROOT_DIR_EVAL, ANNOTATION_FILE_EVAL, transform=get_transform(train=False))
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Model initialization
    model = get_model(NUM_CLASSES).to(DEVICE)
    print("Model initialized successfully.")

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Start training and evaluation
    train_model(model, train_loader, eval_loader, optimizer, DEVICE, NUM_EPOCHS, RESULTS_DIR)

    # Save the final trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Final model saved successfully at {MODEL_PATH}")
