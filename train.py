import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from model import AudioCNN
from dataset import ESC50Dataset
from transform import AudioTransforms
from mixup import Mixup
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, 
                     epoch, num_epochs, writer, use_mixup=True):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=100, file=sys.stdout)
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        
        # Apply mixup with 60% probability
        use_mixup_batch = use_mixup and np.random.random() > 0.4
        
        if use_mixup_batch:
            data, target_a, target_b, lam = Mixup(alpha=0.4).apply(data, target)
            output = model(data)
            loss = Mixup(alpha=0.4).criterion(output, target_a, target_b, lam)
        else:
            output = model(data)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target_a if use_mixup_batch else target).sum().item()
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_val_loss = val_loss / len(val_loader)
    
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auc = 0.0
    
    return avg_val_loss, accuracy, f1, precision, recall, auc


def train(use_cross_validation=False, num_folds=5):
    """Main training function with optional cross-validation"""
    from datetime import datetime
    
    esc50_dir = Path("./ESC-50-master")
    metadata_file = esc50_dir / "meta" / "esc50.csv"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if dataset exists
    if not esc50_dir.exists():
        print(f"ERROR: ESC-50 dataset not found at {esc50_dir}")
        return
    
    print(f"ESC-50 dataset found at {esc50_dir}")
    meta = pd.read_csv(metadata_file)
    classes = sorted(meta['category'].unique())
    print(f"Number of classes: {len(classes)}")
    
    # Get transforms
    train_transform, val_transform = AudioTransforms(sample_rate=22050).get_transforms()

    if use_cross_validation:
        all_accuracies = []
        for fold in range(1, num_folds + 1):
            print(f"\n{'='*60}\nTraining Fold {fold}/{num_folds}\n{'='*60}")
            writer = SummaryWriter(f'./tensorboard_logs/cv_fold{fold}_{timestamp}')
            
            train_folds = [f for f in range(1, num_folds + 1) if f != fold]
            train_dataset = ESC50Dataset(esc50_dir, metadata_file, train_folds=train_folds, transform=train_transform, classes=classes)
            val_dataset = ESC50Dataset(esc50_dir, metadata_file, test_fold=fold, transform=val_transform, classes=classes)
            
            accuracy = train_one_fold(train_dataset, val_dataset, classes, writer, fold, timestamp)
            all_accuracies.append(accuracy)
            writer.close()
        
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results:")
        print(f"Fold accuracies: {[f'{acc:.2f}%' for acc in all_accuracies]}")
        print(f"Mean accuracy: {np.mean(all_accuracies):.2f}%")
        print(f"Std accuracy: {np.std(all_accuracies):.2f}%")
        print(f"{'='*60}")
    
    else:
        writer = SummaryWriter(f'./tensorboard_logs/run_{timestamp}')
        train_dataset = ESC50Dataset(esc50_dir, metadata_file, split="train", transform=train_transform, classes=classes)
        val_dataset = ESC50Dataset(esc50_dir, metadata_file, split="val", transform=val_transform, classes=classes)
        train_one_fold(train_dataset, val_dataset, classes, writer, 0, timestamp)
        writer.close()


def train_one_fold(train_dataset, val_dataset, classes, writer, fold_num, timestamp):
    """Train model on one fold"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(classes)).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    num_epochs = 150
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05, betas=(0.9, 0.999)
)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss, best_accuracy, patience, trigger_times = float('inf'), 0, 30, 0
    
    print(f"\nStarting training on {device}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, num_epochs, writer)
        val_loss, val_acc, f1, precision, recall, auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        Logger.log_metrics(writer, epoch, train_loss, val_loss, train_acc, val_acc, f1, precision, recall, auc, optimizer.param_groups[0]['lr'])
        
        if val_loss < best_val_loss:
            best_val_loss, best_accuracy, trigger_times = val_loss, val_acc, 0
            os.makedirs('./saved_models', exist_ok=True)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss, 'val_accuracy': val_acc,
                        'classes': classes}, 
                       f'./saved_models/best_model_fold{fold_num}_{timestamp}.pth' if fold_num > 0 else f'./saved_models/best_model_{timestamp}.pth')
            print(f'✓ Saved best model: Val Acc {val_acc:.2f}%')
        else:
            trigger_times += 1
            print(f'Validation loss did not improve. Trigger: {trigger_times}/{patience}')
            
            if trigger_times >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    print(f'\nTraining completed! Best validation accuracy: {best_accuracy:.2f}%')

    Logger.plot_confusion_matrix(model, val_loader, classes, fold_num, writer, device)
    return best_accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train ESC-50 Audio Classifier')
    parser.add_argument('--cv', action='store_true', help='Use 5-fold cross-validation')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for CV')
    args = parser.parse_args()
    train(use_cross_validation=args.cv, num_folds=args.folds)
