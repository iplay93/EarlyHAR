import os
import numpy as np
import pandas as pd
import logging


def print_kfold_summary(fold_metrics: dict, early_acc_by_step: dict):
    logging.info("\n=== K-Fold Metric Summary ===")
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        logging.info(f"{metric.capitalize():<10}: {mean_val:.4f} ± {std_val:.4f}")

    logging.info("\n=== K-Fold Early Classification Summary ===")
    for step in sorted(early_acc_by_step.keys()):
        acc_list = early_acc_by_step[step]
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        logging.info(f"Step {int(step*100):>3}% → {mean_acc:.4f} ± {std_acc:.4f}")

def save_kfold_summary_to_csv(
    dataset_name: str,
    fold_metrics: dict,
    early_acc_by_step: dict,
    early_acc_by_fold: list,
    output_dir: str = "results"
):
    # Create dataset-specific result directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. Per-fold metrics (including early accuracy)
    per_fold_data = {
        "fold": list(range(len(fold_metrics["accuracy"]))),
        "accuracy": fold_metrics["accuracy"],
        "precision": fold_metrics["precision"],
        "recall": fold_metrics["recall"],
        "f1_score": fold_metrics["f1_score"]
    }

    # Add early accuracy for each fold
    if early_acc_by_fold:
        step_keys = sorted(early_acc_by_fold[0].keys())  # assume all folds have same steps
        for step in step_keys:
            per_fold_data[f"early_acc@{int(step * 100)}%"] = [
                fold_result.get(step, np.nan) for fold_result in early_acc_by_fold
            ]

    per_fold_df = pd.DataFrame(per_fold_data)
    per_fold_path = os.path.join(dataset_dir, "per_fold_results.csv")
    per_fold_df.to_csv(per_fold_path, index=False)

    # 2. Overall summary (mean ± std)
    summary = {
        "metric": [],
        "mean": [],
        "std": []
    }

    for metric, values in fold_metrics.items():
        summary["metric"].append(metric)
        summary["mean"].append(np.mean(values))
        summary["std"].append(np.std(values))

    for step in sorted(early_acc_by_step.keys()):
        summary["metric"].append(f"early_acc@{int(step * 100)}%")
        summary["mean"].append(np.mean(early_acc_by_step[step]))
        summary["std"].append(np.std(early_acc_by_step[step]))

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(dataset_dir, "kfold_summary.csv")
    summary_df.to_csv(summary_path, index=False)
