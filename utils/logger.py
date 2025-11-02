import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

class Logger:
    """Handles TensorBoard logging for metrics and confusion matrix."""

    def __init__(self, writer):
        self.writer = writer

    def log_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc, f1, precision, recall, auc, lr):
        """Logs all scalar metrics for each epoch."""
        w = self.writer
        w.add_scalar('Loss/Train', train_loss, epoch)
        w.add_scalar('Loss/Validation', val_loss, epoch)
        w.add_scalar('Accuracy/Train', train_acc, epoch)
        w.add_scalar('Accuracy/Validation', val_acc, epoch)
        w.add_scalar('Learning_Rate', lr, epoch)
        w.add_scalar('Metrics/F1', f1, epoch)
        w.add_scalar('Metrics/Precision', precision, epoch)
        w.add_scalar('Metrics/Recall', recall, epoch)
        w.add_scalar('Metrics/AUC', auc, epoch)

        w.add_scalars('Summary/Metrics', {
            'Train Acc': train_acc,
            'Val Acc': val_acc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'AUC': auc
        }, epoch)

    def plot_confusion_matrix(self, model, val_loader, classes, fold_num, device):
        """Generates and logs a confusion matrix to TensorBoard."""
        print(f"\nGenerating confusion matrix for Fold {fold_num} ...")

        all_preds, all_labels = [], []
        model.eval()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, normalize='true') * 100

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm, annot=False, cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar=True, ax=ax, square=True, linewidths=0.8
        )
        ax.set_title(f'Confusion Matrix (Fold {fold_num}) [%]', fontsize=16, pad=15)
        ax.set_xlabel('Predicted', fontsize=14, labelpad=10)
        ax.set_ylabel('True', fontsize=14, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image = np.array(image).transpose(2, 0, 1)
        self.writer.add_image(f'Confusion_Matrix/Fold_{fold_num}', image, 0)
        buf.close()
        plt.close(fig)

        print("Confusion matrix logged successfully to TensorBoard!")
