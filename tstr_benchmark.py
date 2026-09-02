"""
tstr_benchmark.py — Evaluates synthetic data utility via TSTR and TRTR.

Usage:
    python tstr_benchmark.py --real_train_dir ./dataset --synth_train_dir syn_dataset --real_test_dir data/real_holdout_test \
                             --mode both
"""
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import numpy as np

# Re-use your existing dataset loader
from dataset import PatchDataset

def parse_args():
    p = argparse.ArgumentParser(description="TSTR and TRTR Evaluation Benchmark")
    p.add_argument("--real_train_dir", type=str, required=True, help="Path to real training data")
    p.add_argument("--synth_train_dir", type=str, required=True, help="Path to synthetic training data")
    p.add_argument("--real_test_dir", type=str, required=True, help="Path to real holdout test data")
    p.add_argument("--mode", type=str, choices=['TRTR', 'TSTR', 'both'], default='both')
    
    p.add_argument("--model", type=str, default="resnet18", help="Backbone (resnet18 or resnet50)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def build_classifier(model_name, num_classes, device):
    """Loads a pre-trained ResNet and replaces the final head for our tissue classes."""
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
    # Replace the classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def get_balanced_loader(dataset, batch_size):
    """Creates a balanced dataloader to prevent the classifier from ignoring minority classes."""
    class_counts = np.bincount(dataset.labels, minlength=dataset.num_classes)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset), 
        replacement=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

def train_classifier(model, train_loader, epochs, lr, device, run_name):
    """Standard supervised classification training loop."""
    print(f"\n--- Training {run_name} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[{run_name}] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
        
    return model

def evaluate_classifier(model, test_loader, device, run_name, class_names):
    """Evaluates the classifier and computes standard clinical ML metrics."""
    print(f"\n--- Evaluating {run_name} on Real Holdout Set ---")
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    b_acc = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {b_acc:.4f}")
    print(f"Macro F1 Score:    {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {"accuracy": acc, "balanced_accuracy": b_acc, "f1": f1}

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1. Load the strictly real, unseen holdout test set
    print("Loading Real Test Set...")
    test_dataset = PatchDataset(args.real_test_dir, augment=False, stain_normalize=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_classes = test_dataset.num_classes
    class_names = test_dataset.class_names
    
    results = {}

    # 2. TRTR Baseline (Train on Real, Test on Real)
    if args.mode in ['TRTR', 'both']:
        print("\nLoading Real Training Set (TRTR)...")
        real_train_dataset = PatchDataset(args.real_train_dir, augment=True, stain_normalize=False)
        real_train_loader = get_balanced_loader(real_train_dataset, args.batch_size)
        
        model_trtr = build_classifier(args.model, num_classes, device)
        model_trtr = train_classifier(model_trtr, real_train_loader, args.epochs, args.lr, device, "TRTR")
        
        results['TRTR'] = evaluate_classifier(model_trtr, test_loader, device, "TRTR", class_names)

    # 3. TSTR Evaluation (Train on Synthetic, Test on Real)
    if args.mode in ['TSTR', 'both']:
        print("\nLoading Synthetic Training Set (TSTR)...")
        synth_train_dataset = PatchDataset(args.synth_train_dir, augment=True, stain_normalize=False)
        synth_train_loader = get_balanced_loader(synth_train_dataset, args.batch_size)
        
        model_tstr = build_classifier(args.model, num_classes, device)
        model_tstr = train_classifier(model_tstr, synth_train_loader, args.epochs, args.lr, device, "TSTR")
        
        results['TSTR'] = evaluate_classifier(model_tstr, test_loader, device, "TSTR", class_names)

    # 4. Final Comparison
    if args.mode == 'both':
        print("\n=======================================================")
        print("                 FINAL UTILITY COMPARISON              ")
        print("=======================================================")
        print(f"Metric             | TRTR Baseline | TSTR Pipeline | Gap")
        print("-" * 55)
        for metric in ['accuracy', 'balanced_accuracy', 'f1']:
            trtr_val = results['TRTR'][metric]
            tstr_val = results['TSTR'][metric]
            gap = trtr_val - tstr_val
            
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<18} | {trtr_val:.4f}        | {tstr_val:.4f}        | {gap:.4f}")
        print("=======================================================")

if __name__ == "__main__":
    main()