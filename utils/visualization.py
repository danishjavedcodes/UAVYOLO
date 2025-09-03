import matplotlib.pyplot as plt

def plot_training_history(losses, save_path):
    """
    Plot training history and save the figure.
    Args:
        losses (list): List of total loss values per epoch.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_evaluation_metrics(stats, save_path):
    """
    Plot evaluation metrics and save the figure.
    Args:
        stats (list): COCO evaluation statistics.
        save_path (str): Path to save the plot.
    """
    labels = [
        'mAP@[IoU=0.50:0.95]',
        'mAP@[IoU=0.50]',
        'mAP@[IoU=0.75]',
        'AR@[IoU=0.50:0.95]',
        'AR@[maxDets=1]',
        'AR@[maxDets=10]',
        'AR@[maxDets=100]'
    ]
    values = [stats[i] for i in [0, 1, 2, 8, 6, 7, 8]]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta'])
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def plot_multi_granularity_performance(metrics, save_path="multi_granularity_performance.png"):
    """
    Plot multi-granularity performance metrics.
    Args:
        metrics (dict): Dictionary containing metrics for small, medium, and large objects.
        save_path (str): Path to save the plot.
    """
    categories = list(metrics.keys())
    mAP = [metrics[cat]["mAP"] for cat in categories]
    mAP_50 = [metrics[cat]["mAP_50"] for cat in categories]
    mAP_75 = [metrics[cat]["mAP_75"] for cat in categories]
    recall = [metrics[cat]["Recall"] for cat in categories]

    x = range(len(categories))

    plt.figure(figsize=(12, 8))
    plt.bar(x, mAP, width=0.2, label="mAP@[IoU=0.50:0.95]", color="blue", align="center")
    plt.bar([i + 0.2 for i in x], mAP_50, width=0.2, label="mAP@[IoU=0.50]", color="green", align="center")
    plt.bar([i + 0.4 for i in x], mAP_75, width=0.2, label="mAP@[IoU=0.75]", color="red", align="center")
    plt.plot(x, recall, marker="o", linestyle="--", label="Recall", color="orange")

    plt.xticks([i + 0.2 for i in x], categories)
    plt.ylabel("Score")
    plt.title("YOLO-NR Multi-Granularity Analysis Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Multi-granularity performance plot saved at {save_path}")