import os
import matplotlib.pyplot as plt
from utils.visualization import plot_evaluation_metrics


def calculate_metrics(coco_eval, results_dir="results"):
    """
    Calculate and visualize evaluation metrics.
    Args:
        coco_eval (COCOeval): COCO evaluation object.
        results_dir (str): Directory to save the metrics plot.
    """
    os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist
    stats = coco_eval.stats

    precision = stats[0]  # AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
    map_50 = stats[1]     # AP @[ IoU=0.50      | area=   all | maxDets=100 ]
    map_75 = stats[2]     # AP @[ IoU=0.75       | area=   all | maxDets=100 ]
    recall = stats[8]     # AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]

    print(f"Precision (mAP @ IoU=0.50:0.95): {precision:.4f}")
    print(f"mAP @ IoU=0.50: {map_50:.4f}")
    print(f"mAP @ IoU=0.75: {map_75:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save evaluation metrics plot
    metrics_plot_path = os.path.join(results_dir, "evaluation_metrics.png")
    plot_evaluation_metrics(stats, metrics_plot_path)
    print(f"Evaluation metrics plot saved at {metrics_plot_path}")