import torch
import numpy as np


class Mixup:
    """Class-based Mixup augmentation utility"""
    def __init__(self, alpha=0.4):
        """
        Args:
            alpha (float): Mixup interpolation parameter (default=0.4)
        """
        self.alpha = alpha

    def apply(self, x, y):
        """Apply mixup to input batch (x, y)"""

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def criterion(self, criterion, pred, y_a, y_b, lam):
        """Compute mixup loss"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
