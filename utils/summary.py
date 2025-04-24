import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import logging

def get_label_names(dataset_name):
    label_mappings = {
        "doore": {
            0: "Small talk",
            1: "Studying together",
            2: "Technical discussion",
            3: "Seminar",
        },
        "opportunity": {
            0: "Relaxing",
            1: "Coffee time",
            2: "Early morning",
            3: "Cleanup",
            4: "Sandwich time",
        },
        "casas": {
            0: "Fill dispenser",
            1: "Hang clothes",
            2: "Move furniture",
            3: "Sit & read",
            4: "Water plants",
            5: "Sweep floor",
            6: "Play checkers",
            7: "Set ingredients",
            8: "Set table",
            9: "Read magazine",
            10: "Pay electric bill",
            11: "Picnic food",
            12: "Get dishes",
            13: "Pack supplies",
            14: "Pack food & deliver",
        },
        "aras": {
            0: "Other",
            1: "Going Out",
            2: "Preparing Breakfast",
            3: "Having Breakfast",
            4: "Preparing Lunch",
            5: "Having Lunch",
            6: "Preparing Dinner",
            7: "Having Dinner",
            8: "Washing Dishes",
            9: "Having Snack",
            10: "Sleeping",
            11: "Watching TV",
            12: "Studying",
            13: "Having Shower",
            14: "Toileting",
            15: "Napping",
        },
        "openpack": {
            0: "Picking",
            1: "Relocate item label",
            2: "Assemble box",
            3: "Insert items",
            4: "Close box",
            5: "Attach box label",
            6: "Scan label",
            7: "Attach shipping label",
            8: "Put on back table",
            9: "Fill out order",
        }
    }
    return label_mappings.get(dataset_name, {})

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

def plot_classwise_early_accuracy_with_std(dataset_name, csv_path, save_path=None):
    """
    Plots class-wise early classification accuracy with optional std shading.

    Args:
        dataset_name (str): Dataset name to resolve activity labels.
        csv_path (str): Path to CSV file with columns ['step', 'class', 'accuracy'].
        save_path (str or None): If provided, saves the figure to the path.
    """
    df = pd.read_csv(csv_path)
    label_names = get_label_names(dataset_name)

    # 평균과 표준편차 계산
    mean_df = df.groupby(['step', 'class'])['accuracy'].mean().reset_index()
    std_df = df.groupby(['step', 'class'])['accuracy'].std().reset_index()

    # Merge for easy plotting
    merged = pd.merge(mean_df, std_df, on=['step', 'class'], suffixes=('_mean', '_std'))

    # 각 클래스별로 라인 + 면적 시각화
    plt.figure(figsize=(10, 6))
    classes = merged['class'].unique()

    for cls in classes:
        class_data = merged[merged['class'] == cls].sort_values('step')
        steps = class_data['step']
        acc_mean = class_data['accuracy_mean']
        #acc_std = class_data['accuracy_std']

        label = label_names.get(cls, f"Class {cls}")
        plt.plot(steps, acc_mean, label=label)
        #plt.fill_between(steps, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)

    plt.xlabel("Time Step (%)")
    plt.ylabel("Accuracy")
    plt.legend(title="Activity")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def save_kfold_summary_to_csv(
    dataset_name: str,
    fold_metrics: dict,
    early_acc_by_step: dict,
    early_acc_by_fold: list,
    early_classwise_acc_by_fold: list, 
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

    # 3. Class-wise Early Accuracy (per fold and step)
    classwise_records = []

    for fold_idx, fold_classwise in enumerate(early_classwise_acc_by_fold):
        for step, class_acc_dict in fold_classwise.items():
            for cls, acc in class_acc_dict.items():
                classwise_records.append({
                    "fold": fold_idx,
                    "step": int(step * 100),
                    "class": cls,
                    "accuracy": acc
                })

    if classwise_records:
        classwise_df = pd.DataFrame(classwise_records)
        classwise_path = os.path.join(dataset_dir, "classwise_early_accuracy.csv")
        classwise_df.to_csv(classwise_path, index=False)

        plot_classwise_early_accuracy_with_std(
            dataset_name,
            classwise_path,
            save_path= os.path.join(dataset_dir, "classwise_early_plot.png")
        )
