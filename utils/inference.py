import time
import torch
def measure_inference_time(model, data_loader, device):
    """
    Measure the average inference time per image.
    Args:
        model: Trained detection model.
        data_loader: DataLoader for evaluation dataset.
        device: Device (CPU/GPU) to run the model on.
    """
    model.eval()
    total_time = 0
    num_images = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = list(image.to(device) for image in images)
            start_time = time.time()
            _ = model(images)
            end_time = time.time()
            total_time += (end_time - start_time)
            num_images += len(images)

    avg_inference_time = total_time / num_images
    print(f"Average Inference Time per Image: {avg_inference_time:.4f} seconds")