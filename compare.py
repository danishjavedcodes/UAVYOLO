import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# For silent prints
from contextlib import redirect_stdout

# Try to silence ultralytics logs via logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from ultralytics import YOLO

# Additional object detection models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

from models.faster_rcnn import get_model
from utils.transforms import get_transform
from utils.inference import measure_inference_time



def run_silently(func, *args, **kwargs):
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            return func(*args, **kwargs)


#                     YOLO EVALUATION 

def evaluate_yolo_model(ckpt_path, data_yaml="./data/SAR/YOLO/data.yaml", model_label="YOLO"):
    model = YOLO(ckpt_path)
    results = run_silently(model.val, data=data_yaml, verbose=False)

    precision = results.box.mp       # mean precision
    recall = results.box.mr          # mean recall
    map50 = results.box.map50        # mAP@0.5
    map50_95 = results.box.map       # mAP@0.5:0.95

    return {
        "model_name": model_label,
        "precision": float(precision),
        "recall": float(recall),
        "mAP50": float(map50),
        "mAP50_95": float(map50_95)
    }


#                 ADDITIONAL OBJECT DETECTION MODELS

def get_faster_rcnn_model(num_classes):
    """Create a standard Faster R-CNN model"""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_retinanet_model(num_classes):
    """Create a standard RetinaNet model"""
    model = retinanet_resnet50_fpn(weights="DEFAULT")
    # RetinaNet doesn't need modification for num_classes as it's handled differently
    return model


def evaluate_torchvision_model(model, dataset_root, annotation_file,
                              num_classes=3, batch_size=4, device=None,
                              model_label="TorchVision Model"):
    """Generic evaluation function for torchvision models"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset/dataloader
    eval_dataset = CustomDataset(dataset_root, annotation_file, transform=get_transform(train=False))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Move model to device
    model.to(device)
    model.eval()

    coco_gt = COCO(annotation_file)
    results = []

    # Run inference silently
    with torch.no_grad():
        for images, targets in eval_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                try:
                    image_id = targets[i]["image_id"].item()
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
                        "score": float(score),
                    })

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    run_silently(silent_coco_eval, coco_eval)  # fully silent

    stats = coco_eval.stats 

    return {
        "model_name": model_label,
        "precision": float(stats[1]),   # AP@0.5
        "recall": float(stats[8]),      # AR@0.5:0.95
        "mAP50": float(stats[1]),       # AP@0.5
        "mAP50_95": float(stats[0])     # AP@0.5:0.95
    }



#                  CUSTOM MODEL EVALUATION                

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = int(self.image_ids[idx])
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2]
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.uint8)
        }

        if self.transform and callable(self.transform):
            image = self.transform(image)

        return image, target


def silent_coco_eval(coco_eval):
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()


def evaluate_custom_model(model, dataset_root, annotation_file,
                          num_classes=72, batch_size=4, device=None,
                          model_label="Yolo NR"):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset/dataloader
    eval_dataset = CustomDataset(dataset_root, annotation_file, transform=get_transform(train=False))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Move model to device
    model.to(device)
    model.eval()

    coco_gt = COCO(annotation_file)
    results = []

    # Run inference silently
    with torch.no_grad():
        for images, targets in eval_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                try:
                    image_id = targets[i]["image_id"].item()
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
                        "score": float(score),
                    })

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    run_silently(silent_coco_eval, coco_eval)  # fully silent

    stats = coco_eval.stats 

    return {
        "model_name": model_label,
        "precision": float(stats[1]),   # AP@0.5
        "recall": float(stats[8]),      # AR@0.5:0.95
        "mAP50": float(stats[1]),       # AP@0.5
        "mAP50_95": float(stats[0])     # AP@0.5:0.95
    }



#                 PRINTING & PLOTTING RESULTS             

def print_metrics_table(results_list):
    """
    Print final metrics in a table-like format, no extra logs.
    """
    header = (
        f"{'Model':<15} | {'Precision':>10} | {'Recall':>10} | "
        f"{'mAP@0.5':>10} | {'mAP@0.5:0.95':>13}"
    )
    print(header)
    print("-" * len(header))
    for res in results_list:
        print(
            f"{res['model_name']:<15} | "
            f"{res['precision']:.4f}{'':>3} | "
            f"{res['recall']:.4f}{'':>5} | "
            f"{res['mAP50']:.4f}{'':>4} | "
            f"{res['mAP50_95']:.4f}"
        )


def plot_bar_chart(results_list, output_path="comparison.png"):
    # Metrics to compare
    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    model_names = [res["model_name"] for res in results_list]

    # Convert data to 2D array: rows=metrics, columns=models
    data = []
    for metric in metrics:
        data.append([res[metric] for res in results_list])
    data = np.array(data)  # shape=(4, n_models)

    x = np.arange(len(metrics))  # [0,1,2,3] for 4 metrics
    total_models = len(results_list)
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot each model as a separate offset
    for i in range(total_models):
        ax.bar(
            x + i * bar_width,
            data[:, i],
            bar_width,
            label=model_names[i]
        )

    ax.set_ylabel("Score")
    ax.set_xticks(x + bar_width * (total_models - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    # Save the figure, do NOT display it
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to avoid display



#                        MAIN SCRIPT  

if __name__ == "__main__":
    # 1. Evaluate YOLO v11
    yolo_v11_results = evaluate_yolo_model(
        ckpt_path="v11.pt",
        data_yaml="./data/SAR/YOLO/data.yaml",
        model_label="YOLOv11"
    )

    # 2. Evaluate YOLO v10
    yolo_v10_results = evaluate_yolo_model(
        ckpt_path="v10.pt",
        data_yaml="./data/SAR/YOLO/data.yaml",
        model_label="YOLOv10"
    )

    # 3. Evaluate Custom Model (Yolo NR)
    ROOT_DIR_EVAL = "./data/SAR/valid"
    ANNOTATION_FILE_EVAL = "./data/SAR/valid/_annotations.coco.json"
    MODEL_PATH = "./object_detection_model_SAR_data.pth"

    NUM_CLASSES = 3
    BATCH_SIZE = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    custom_model = get_model(NUM_CLASSES)
    custom_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    custom_model_results = evaluate_custom_model(
        model=custom_model,
        dataset_root=ROOT_DIR_EVAL,
        annotation_file=ANNOTATION_FILE_EVAL,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        model_label="Yolo NR"
    )

    # 4. Evaluate Faster R-CNN
    faster_rcnn_model = get_faster_rcnn_model(NUM_CLASSES)
    faster_rcnn_results = evaluate_torchvision_model(
        model=faster_rcnn_model,
        dataset_root=ROOT_DIR_EVAL,
        annotation_file=ANNOTATION_FILE_EVAL,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        model_label="Faster R-CNN"
    )

    # 5. Evaluate RetinaNet
    retinanet_model = get_retinanet_model(NUM_CLASSES)
    retinanet_results = evaluate_torchvision_model(
        model=retinanet_model,
        dataset_root=ROOT_DIR_EVAL,
        annotation_file=ANNOTATION_FILE_EVAL,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        model_label="RetinaNet"
    )

    # Measure inference time for custom model
    dataset_for_time = CustomDataset(
        root_dir=ROOT_DIR_EVAL,
        annotation_file=ANNOTATION_FILE_EVAL,
        transform=get_transform(train=False)
    )
    loader_for_time = DataLoader(
        dataset_for_time,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    run_silently(measure_inference_time, custom_model, loader_for_time, DEVICE)

    # Combine all results
    all_results = [
        yolo_v11_results,
        yolo_v10_results,
        custom_model_results,
        faster_rcnn_results,
        retinanet_results
    ]

    # Print only final comparison
    print("\n--- Model Comparison Table ---")
    print_metrics_table(all_results)

    # Save comparison bar chart, do not show
    plot_bar_chart(all_results, output_path="model_comparison.png")

    print("\nBar chart saved to 'model_comparison.png'")
