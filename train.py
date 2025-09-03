import os
import torch
from torch.utils.data import DataLoader
from models.faster_rcnn import get_hmaytsf_baseline_model as get_model
from datasets.custom_dataset import CustomDataset
from utils.transforms import get_transform
from utils.visualization import plot_training_history
from tqdm import tqdm



def train_model(model, train_loader, eval_loader, optimizer, device, num_epochs, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
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


        # Save checkpoint
        checkpoint_path = os.path.join(results_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved at: {checkpoint_path}')

