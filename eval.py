import os
import torch
from torch.utils.data import DataLoader
from PIL import Image  # Import the Image module from Pillow <button class="citation-flag" data-index="3">
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# Import custom modules
from models.faster_rcnn import get_hmaytsf_baseline_model as get_model
from utils.transforms import get_transform
from utils.inference import measure_inference_time

# Paths
ROOT_DIR_EVAL = "./Aerial-Vehicles-1/valid"
ANNOTATION_FILE_EVAL = "./Aerial-Vehicles-1/valid/_annotations.coco.json"
MODEL_PATH = "/workspace/UAVYOLO/results/model_epoch_1.pth"
# MODEL_PATH = "/home/danish/Desktop/Danish's/YOLO_NR/results/model_epoch_21.pth"

# Hyperparameters
NUM_CLASSES = 5  # Update based on your dataset (e.g., 71 categories + background)
BATCH_SIZE = 1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Custom Dataset Implementation
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
        image = Image.open(image_path).convert("RGB")  # Use the imported Image module <button class="citation-flag" data-index="3">

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2] format
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

        # Apply transformations
        if self.transform:
            if callable(self.transform):  # Ensure transform is callable
                image = self.transform(image)  # Apply transformation to the image only
            else:
                raise ValueError("Transform must be a callable function.")

        return image, target

# Evaluation Function
def evaluate_model(model, data_loader, device):
    coco_gt = COCO(ANNOTATION_FILE_EVAL)  # Ground truth annotations
    results = []

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for images, targets in data_loader:

            # Ensure images are on the correct device
            images = list(image.to(device) for image in images)
            outputs = model(images)  # Perform inference

            # Iterate over predictions and ground truth
            for i, output in enumerate(outputs):
                try:
                    # Ensure targets[i] is a dictionary
                    if isinstance(targets[i], dict):
                        image_id = targets[i]["image_id"].item()
                    else:
                        raise TypeError(f"Unexpected type for targets[{i}]: {type(targets[i])}")
                except (KeyError, TypeError) as e:
                    print(f"Error processing targets[{i}]: {e}")
                    print("Using batch index as fallback for image_id.")
                    image_id = i  # Fallback to batch index

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # COCO format: [x, y, w, h]
                        "score": score
                    })

    # Save predictions to COCO format
    coco_dt = coco_gt.loadRes(results)

    # Evaluate using COCO API
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats  # Returns mAP and other metrics

# Main Script
if __name__ == "__main__":
    # Create datasets and dataloaders for evaluation
    eval_dataset = CustomDataset(ROOT_DIR_EVAL, ANNOTATION_FILE_EVAL, transform=get_transform(train=False))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))  # Ensure targets remain a list of dictionaries
    )

    # Load the trained model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))  # Address FutureWarning <button class="citation-flag" data-index="2">
    model.to(DEVICE)
    print("Trained model loaded successfully.")

    # Run evaluation
    print("Evaluating the model...")
    metrics = evaluate_model(model, eval_loader, DEVICE)

    # Print evaluation metrics
    print(f"mAP (IoU=0.50:0.95): {metrics[0]:.4f}")
    print(f"mAP (IoU=0.50): {metrics[1]:.4f}")
    print(f"mAP (IoU=0.75): {metrics[2]:.4f}")

    measure_inference_time(model, eval_loader, DEVICE)