import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_model(model, test_loader, save_path=None, device='cuda', cm_image_path='results/confusion_matrix.png'):
    # Load saved model state
    if save_path is not None:
        print(f"Loading model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("No save path provided. Using current model state.")

    model.to(device)
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(y_batch.numpy())

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"\n📊 Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=[f"Class {l}" for l in sorted(set(labels))]))
    print("\nConfusion Matrix:")
    print(cm)

    # === Normalize confusion matrix ===
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

    # Ensure save directory exists
    os.makedirs(os.path.dirname(cm_image_path), exist_ok=True)

    # === Plot and Save Normalized Confusion Matrix ===
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=[f"{l+1}" for l in sorted(set(labels))],
                yticklabels=[f"{l+1}" for l in sorted(set(labels))])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_image_path, dpi=300)
    plt.close()
    print(f"Normalized confusion matrix saved: {cm_image_path}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm_normalized
    }
