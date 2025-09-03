import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.image_info = self.annotations['images']
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        self.image_id_to_anns = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        annotations = self.image_id_to_anns.get(img_info['id'], [])
        boxes, labels = [], []
        for ann in annotations:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        if self.transform:
            image = self.transform(image)
        return image, target