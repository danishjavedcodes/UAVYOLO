import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from models.faster_rcnn import get_model
from utils.transforms import get_transform
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import json
# Additional imports for more object detection models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class ResultVisualizer:
    def __init__(self, dataset_root, annotation_file):
        self.dataset_root = dataset_root
        self.coco = COCO(annotation_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create class mapping for COCO models
        self.class_mapping = {}
        for cat_id, cat_info in self.coco.cats.items():
            # Map COCO class IDs to our dataset class IDs
            if cat_info['name'] == 'ship':
                self.class_mapping[cat_id] = cat_id  # Keep the same ID
        
    def load_models(self):
        # Load YOLO models
        self.yolo_v11 = YOLO("./v11.pt")
        self.yolo_v10 = YOLO("./v10.pt")
        
        # Load custom model (YOLO NR)
        self.custom_model = get_model(num_classes=3)
        self.custom_model.load_state_dict(
            torch.load("./object_detection_model_SAR_data.pth", map_location=self.device)
        )
        self.custom_model.to(self.device)
        self.custom_model.eval()
        
        # Load standard Faster R-CNN model
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        self.faster_rcnn.to(self.device)
        self.faster_rcnn.eval()
        
        # Load RetinaNet model
        self.retinanet = retinanet_resnet50_fpn(weights="DEFAULT")
        self.retinanet.to(self.device)
        self.retinanet.eval()

    def get_predictions(self, image_path):
        # YOLO v11 predictions
        results_v11 = self.yolo_v11(image_path)
        boxes_v11 = results_v11[0].boxes.xyxy.cpu().numpy()
        classes_v11 = results_v11[0].boxes.cls.cpu().numpy()
        conf_v11 = results_v11[0].boxes.conf.cpu().numpy()

        # YOLO v10 predictions
        results_v10 = self.yolo_v10(image_path)
        boxes_v10 = results_v10[0].boxes.xyxy.cpu().numpy()
        classes_v10 = results_v10[0].boxes.cls.cpu().numpy()
        conf_v10 = results_v10[0].boxes.conf.cpu().numpy()

        # Custom model predictions (YOLO NR)
        image = Image.open(image_path).convert("RGB")
        transform = get_transform(train=False)
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.custom_model(image_tensor)[0]
        boxes_custom = predictions['boxes'].cpu().numpy()
        classes_custom = predictions['labels'].cpu().numpy()
        conf_custom = predictions['scores'].cpu().numpy()
        
        # Faster R-CNN predictions
        with torch.no_grad():
            predictions_faster_rcnn = self.faster_rcnn(image_tensor)[0]
        boxes_faster_rcnn = predictions_faster_rcnn['boxes'].cpu().numpy()
        classes_faster_rcnn = predictions_faster_rcnn['labels'].cpu().numpy()
        conf_faster_rcnn = predictions_faster_rcnn['scores'].cpu().numpy()
        
        # RetinaNet predictions
        with torch.no_grad():
            predictions_retinanet = self.retinanet(image_tensor)[0]
        boxes_retinanet = predictions_retinanet['boxes'].cpu().numpy()
        classes_retinanet = predictions_retinanet['labels'].cpu().numpy()
        conf_retinanet = predictions_retinanet['scores'].cpu().numpy()

        return (boxes_v11, classes_v11, conf_v11), (boxes_v10, classes_v10, conf_v10), (boxes_custom, classes_custom, conf_custom), (boxes_faster_rcnn, classes_faster_rcnn, conf_faster_rcnn), (boxes_retinanet, classes_retinanet, conf_retinanet)

    def get_ground_truth(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        boxes = []
        classes = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            classes.append(ann['category_id'])
        return np.array(boxes), np.array(classes)

    def get_class_name(self, class_id):
        """Safely get class name from class ID"""
        if class_id in self.coco.cats:
            return self.coco.cats[class_id]["name"]
        else:
            return f"class_{class_id}"

    # In plot_multiple_results method, update the results storage:
    def plot_results(self, image_path, image_id, output_path):
        # Get ground truth and predictions
        gt_boxes = self.get_ground_truth(image_id)
        boxes_v11, boxes_v10, boxes_custom, boxes_faster_rcnn, boxes_retinanet = self.get_predictions(image_path)

        # Create subplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Predictions Comparison', fontsize=16)

        # Load and convert image
        image = plt.imread(image_path)
        
        # Plot ground truth
        axes[0, 0].imshow(image)
        self.draw_boxes(axes[0, 0], gt_boxes, 'Ground Truth', 'green')

        # Plot YOLO v11 predictions
        axes[0, 1].imshow(image)
        self.draw_boxes(axes[0, 1], boxes_v11, 'YOLO v11', 'blue')

        # Plot YOLO v10 predictions
        axes[0, 2].imshow(image)
        self.draw_boxes(axes[0, 2], boxes_v10, 'YOLO v10', 'red')

        # Plot Custom model predictions
        axes[1, 0].imshow(image)
        self.draw_boxes(axes[1, 0], boxes_custom, 'YOLO NR', 'purple')

        # Plot Faster R-CNN predictions
        axes[1, 1].imshow(image)
        self.draw_boxes(axes[1, 1], boxes_faster_rcnn, 'Faster R-CNN', 'orange')

        # Plot RetinaNet predictions
        axes[1, 2].imshow(image)
        self.draw_boxes(axes[1, 2], boxes_retinanet, 'RetinaNet', 'brown')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def draw_boxes(self, ax, boxes, title, color):
        ax.set_title(title)
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
        ax.axis('off')

    def plot_multiple_results(self, num_images=3):
        specific_images = [
            "./train/000000000357_jpg.rf.fd60a5947f0f1b2ad6273d8ff87b6282.jpg",
            "./train/000000000419_jpg.rf.b58c3291ae18177c82e19195fd533614.jpg",
            "./train/000000000520_jpg.rf.30218d72a576772917cc478208e40924.jpg"
        ]
        
        # Create figure with a grid layout
        fig = plt.figure(figsize=(30, 5*len(specific_images)))
        
        # Create grid specification for layout with extra column for sample numbers
        gs = fig.add_gridspec(len(specific_images) + 1, 7, 
                             width_ratios=[0.2, 1, 1, 1, 1, 1, 1],
                             height_ratios=[0.2] + [1]*len(specific_images))
        
        # Add column headers (skip first column for sample numbers)
        headers = ['Ground Truth', 'YOLO v10', 'YOLO v11', 'YOLO NR', 'Faster R-CNN', 'RetinaNet']
        for col, header in enumerate(headers):
            ax = fig.add_subplot(gs[0, col+1])
            ax.text(0.5, 0.5, header, ha='center', va='center', fontsize=12, fontweight='bold')
            ax.axis('off')
    
        # Create dictionary to store results
        results_data = {
            "model_comparison": {
                "models": ["Ground Truth", "YOLO v10", "YOLO v11", "YOLO NR", "Faster R-CNN", "RetinaNet"],
                "samples": []
            }
        }

        for row, image_path in enumerate(specific_images):
            # Add sample number in the first column
            ax_sample = fig.add_subplot(gs[row + 1, 0])
            ax_sample.text(0.5, 0.5, f'Sample {row + 1}', 
                         ha='center', va='center', 
                         fontsize=12, fontweight='bold',
                         rotation=0)
            ax_sample.axis('off')
            
            # Get image ID and predictions
            image_filename = os.path.basename(image_path)
            image_id = next(
                (k for k, v in self.coco.imgs.items() 
                 if v['file_name'] == image_filename), None
            )
            
            if image_id is None:
                print(f"Warning: Could not find image {image_filename} in annotations")
                continue
            
            gt_boxes, gt_classes = self.get_ground_truth(image_id)  # Unpack the tuple
            (boxes_v11, classes_v11, conf_v11), (boxes_v10, classes_v10, conf_v10), (boxes_custom, classes_custom, conf_custom), (boxes_faster_rcnn, classes_faster_rcnn, conf_faster_rcnn), (boxes_retinanet, classes_retinanet, conf_retinanet) = self.get_predictions(image_path)
            image = plt.imread(image_path)
            
            # Plot each cell in the row (shifted by 1 column)
            boxes_list = [gt_boxes, boxes_v10, boxes_v11, boxes_custom, boxes_faster_rcnn, boxes_retinanet]
            colors = ['green', 'red', 'blue', 'purple', 'orange', 'brown']
            
            for col in range(6):
                ax = fig.add_subplot(gs[row + 1, col + 1])
                ax.imshow(image)
                self.draw_boxes(ax, boxes_list[col], '', colors[col])

            # Store results for this sample with correct unpacking
            sample_data = {
                "sample_number": row + 1,
                "image_name": image_filename,
                "detections": {
                    "ground_truth": {
                        "boxes": gt_boxes.tolist(),
                        "classes": gt_classes.tolist(),
                        "class_names": [self.get_class_name(cls_id) for cls_id in gt_classes]
                    },
                    "yolo_v10": {
                        "boxes": boxes_v10.tolist(),
                        "classes": classes_v10.tolist(),
                        "confidence": conf_v10.tolist(),
                        "class_names": [self.yolo_v10.names[int(cls_id)] for cls_id in classes_v10]
                    },
                    "yolo_v11": {
                        "boxes": boxes_v11.tolist(),
                        "classes": classes_v11.tolist(),
                        "confidence": conf_v11.tolist(),
                        "class_names": [self.yolo_v11.names[int(cls_id)] for cls_id in classes_v11]
                    },
                    "yolo_nr": {
                        "boxes": boxes_custom.tolist(),
                        "classes": classes_custom.tolist(),
                        "confidence": conf_custom.tolist(),
                        "class_names": [self.get_class_name(cls_id) for cls_id in classes_custom]
                    },
                    "faster_rcnn": {
                        "boxes": boxes_faster_rcnn.tolist(),
                        "classes": classes_faster_rcnn.tolist(),
                        "confidence": conf_faster_rcnn.tolist(),
                        "class_names": [self.get_class_name(cls_id) for cls_id in classes_faster_rcnn]
                    },
                    "retinanet": {
                        "boxes": boxes_retinanet.tolist(),
                        "classes": classes_retinanet.tolist(),
                        "confidence": conf_retinanet.tolist(),
                        "class_names": [self.get_class_name(cls_id) for cls_id in classes_retinanet]
                    }
                }
            }
            results_data["model_comparison"]["samples"].append(sample_data)

            print(f"Processed image {row + 1}/{len(specific_images)}: {image_filename}")
    
        # Save results to JSON file
        json_output_path = "./results/detection_results.json"
        with open(json_output_path, 'w') as f:
            json.dump(results_data, f, indent=4)  # Changed from json.dumps to json.dump
        print(f"Saved detailed results to {json_output_path}")
        plt.tight_layout()
        output_path = "./results/all_comparisons.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved all comparisons to {output_path}")

    def draw_boxes(self, ax, boxes, title, color):
        ax.set_title(title, pad=10)  # Add padding to title
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
        ax.axis('off')

# Move main() function outside the class
def main():
    dataset_root = "./data/SAR/valid"
    annotation_file = "./data/SAR/valid/_annotations.coco.json"
    
    visualizer = ResultVisualizer(dataset_root, annotation_file)
    visualizer.load_models()
    visualizer.plot_multiple_results(num_images=3)

if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    main()