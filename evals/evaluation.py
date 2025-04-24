import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging, wandb
from collections import defaultdict

def evaluate_model(model, test_loader, args):
    """
    Evaluate model performance using test_loader and log results.
    Generates and saves normalized confusion matrix.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader with test data.
        args: Argument object containing device, save_path, cm_image_path, etc.
    """

    device = args.device
    save_path = args.save_path
    cm_image_path = args.cm_image_path

    # # Load saved model state
    # if save_path is not None:
    #     logging.info(f"Loading model from {save_path}")
    #     model.load_state_dict(torch.load(save_path, map_location=device))
    # else:
    #     logging.warning("No save path provided. Using current model state.")

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

    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logging.info("Classification Report:")
    logging.info("\n" + classification_report(labels, preds, target_names=[f"Class {l}" for l in sorted(set(labels))]))
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")

    # === Normalize confusion matrix ===
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

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
    logging.info(f"Normalized confusion matrix saved: {cm_image_path}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm_normalized
    }

def evaluate_early_classification(model, test_loader, args, step_interval=0.1):
    model.eval()
    device = args.device
    results_by_step = {}
    classwise_results_by_step = defaultdict(dict)  # [step][class] = acc

    steps = np.arange(step_interval, 1.0 + step_interval, step_interval)

    for step in steps:
        preds = []
        labels = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                seq_len = x_batch.shape[1]
                partial_len = max(1, int(seq_len * step))
                x_partial = x_batch[:, :partial_len, :]

                if partial_len < seq_len:
                    padding_len = seq_len - partial_len
                    pad_tensor = torch.zeros(x_partial.shape[0], padding_len, x_partial.shape[2], device=device)
                    x_partial_padded = torch.cat([x_partial, pad_tensor], dim=1)
                else:
                    x_partial_padded = x_partial

                outputs = model(x_partial_padded)
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                labels.extend(y_batch.cpu().numpy())

        step_key = round(step, 2)
        acc = accuracy_score(labels, preds)
        results_by_step[step_key] = acc
        logging.info(f"[Early Classification] Step: {step:.2f} | Accuracy: {acc:.4f}")

        # === calcluate class-wise performance ===
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        for class_label in report:
            if class_label.isdigit():  # class-wise entries only
                classwise_results_by_step[step_key][int(class_label)] = report[class_label]["precision"]

    # === wandb Table → Line Plot ===
    table = wandb.Table(columns=["Step", "Accuracy"])
    for step, acc in results_by_step.items():
        table.add_data(int(step * 100), acc)

    wandb.log({
        "Early Classification Accuracy": wandb.plot.line(
            table, "Step", "Accuracy", title="Early Classification Accuracy"
        )
    })

    return results_by_step, classwise_results_by_step
