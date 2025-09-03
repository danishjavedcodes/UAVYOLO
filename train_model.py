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
from models.faster_rcnn import get_hmaytsf_baseline_model as  get_model
from datasets.custom_dataset import CustomDataset
from utils.transforms import get_transform
from train import train_model


if __name__ == "__main__":
    # Paths
    ROOT_DIR_TRAIN = "./Aerial-Vehicles-1/train"
    ANNOTATION_FILE_TRAIN = "./Aerial-Vehicles-1/train/_annotations.coco.json"
    ROOT_DIR_EVAL = "./Aerial-Vehicles-1/valid"
    ANNOTATION_FILE_EVAL = "./Aerial-Vehicles-1/valid/_annotations.coco.json"
    MODEL_PATH = "/home/danish/Desktop/Danish's/YOLO_NR/object_detection_model_SAR_data.pth"
    RESULTS_DIR = "results"

    # Hyperparameters
    NUM_CLASSES = 5
    BATCH_SIZE = 16
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.005
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {DEVICE}')

    # Training dataset and loader
    train_dataset = CustomDataset(ROOT_DIR_TRAIN, ANNOTATION_FILE_TRAIN, transform=get_transform(train=True))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Evaluation dataset and loader (use batch_size=1 for proper COCO evaluation)
    eval_dataset = CustomDataset(ROOT_DIR_EVAL, ANNOTATION_FILE_EVAL, transform=get_transform(train=False))
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Model initialization
    model = get_model(NUM_CLASSES).to(DEVICE)
    print("Model initialized successfully.")

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Start training and evaluation
    train_model(model, train_loader, eval_loader, optimizer, DEVICE, NUM_EPOCHS, RESULTS_DIR, ANNOTATION_FILE_EVAL)

    # Save the final trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Final model saved successfully at {MODEL_PATH}")
