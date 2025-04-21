import pickle
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from baselines.calimera import CALIMERA
from data_preprocessing.data_preprocess import pad_sequences, balance_by_augmentation
from utils.summary import print_kfold_summary, save_kfold_summary_to_csv
import os
import yaml

import argparse

def load_yaml_config(dataset_name):
    config_path = os.path.join('configs', f'{dataset_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_args():
    parser = argparse.ArgumentParser()

    # --- General Arguments ---
    parser.add_argument('--dataset', type=str, default='doore')
    parser.add_argument('--padding', type=str, default='mean')
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--aug_method', type=str, default='noise')

    # --- CALIMERA Specific ---
    parser.add_argument('--delay_penalty', type=int, default=1)
    parser.add_argument('--k_fold', type=int, default=5)

    # --- Parse ---
    args = parser.parse_args()

    # --- Dataset Config ---
    dataset_config = load_yaml_config(args.dataset)
    for key, value in dataset_config.items():
        setattr(args, key, value)

    # --- Paths ---
    args.save_dir = getattr(args, 'save_dir', 'save_model')
    os.makedirs(args.save_dir, exist_ok=True)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args

def f_e_metric(accuracy, earliness):
    return 2 * ((1 - earliness) * accuracy) / ((1 - earliness) + accuracy + 1e-8)

if __name__ == '__main__':

    args = setup_args()

    fold_metrics = {
        "accuracy": [],
        "earliness": [],
        "cost": [],
        "f_e": []
    }
    per_fold_results = []

    for fold_idx in range(args.k_fold):
        print(f"\n=== Fold {fold_idx} ===")

        # Load data
        fold_path = f'fold_data/{args.dataset}/fold_{fold_idx}.pkl'
        with open(fold_path, 'rb') as f:
            fold_data = pickle.load(f)

        train_data = fold_data['train_data']
        train_labels = fold_data['train_labels']
        test_data = fold_data['test_data']
        test_labels = fold_data['test_labels']

        # Augmentation
        if args.augment:
            print(f"[Augment] Method: {args.aug_method}")
            train_data, train_labels = balance_by_augmentation(train_data, train_labels, method=args.aug_method)

        # Padding
        train_tensor, _ = pad_sequences(train_data, padding_type=args.padding)
        test_tensor, _ = pad_sequences(test_data, padding_type=args.padding)

        X_train = train_tensor.permute(0, 2, 1).numpy()
        X_test = test_tensor.permute(0, 2, 1).numpy()

        y_train = train_labels.numpy() if isinstance(train_labels, torch.Tensor) else np.array(train_labels)
        y_test = test_labels.numpy() if isinstance(test_labels, torch.Tensor) else np.array(test_labels)

        # Fit and Evaluate CALIMERA
        model = CALIMERA(delay_penalty=args.delay_penalty)
        model.fit(X_train, y_train)
        stop_timestamps, y_pred = model.test(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        earliness = sum(stop_timestamps) / (X_test.shape[-1] * X_test.shape[0])
        cost = 1.0 - accuracy + args.delay_penalty * earliness
        f_e = f_e_metric(accuracy, earliness)

        print(f"Accuracy: {accuracy:.4f} | Earliness: {earliness:.4f} | Cost: {cost:.4f} | F-E: {f_e:.4f}")

        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["earliness"].append(earliness)
        fold_metrics["cost"].append(cost)
        fold_metrics["f_e"].append(f_e)

        per_fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy,
            "earliness": earliness,
            "cost": cost,
            "f_e": f_e
        })

    # Summary
    print("\n=== K-Fold Summary ===")
    for key in fold_metrics:
        values = np.array(fold_metrics[key])
        print(f"{key.upper()}: {values.mean():.4f} ± {values.std():.4f}")

    save_kfold_summary_to_csv(
        args.dataset,
        metric_dict=fold_metrics,
        early_acc_by_step={},  # optional
        early_acc_by_fold=per_fold_results,
        filename=f"results/calimera_{args.dataset}_summary.csv"
    )
