import pickle
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from baselines.calimera import CALIMERA
from data_preprocessing.data_preprocess import pad_sequences, balance_by_augmentation

def load_example_data(dataset='doore',
                      fold_idx=0,
                      padding_type='mean',
                      augment=False,
                      aug_method='noise'):
    fold_path = f'fold_data/{dataset}/fold_{fold_idx}.pkl'

    # Load saved fold data
    with open(fold_path, 'rb') as f:
        fold_data = pickle.load(f)

    train_data = fold_data['train_data']
    train_labels = fold_data['train_labels']
    test_data = fold_data['test_data']
    test_labels = fold_data['test_labels']

    # Apply augmentation to training data only
    if augment:
        print(f"[Augment] Method: {aug_method}")
        train_data, train_labels = balance_by_augmentation(train_data, train_labels, method=aug_method)

    # Apply padding (same as in load_fold_data)
    train_tensor, _ = pad_sequences(train_data, padding_type=padding_type)
    test_tensor, _ = pad_sequences(test_data, padding_type=padding_type)

    # Convert to [N, C, T]
    X_train = train_tensor.permute(0, 2, 1).numpy()
    X_test = test_tensor.permute(0, 2, 1).numpy()

    y_train = train_labels.numpy() if isinstance(train_labels, torch.Tensor) else np.array(train_labels)
    y_test = test_labels.numpy() if isinstance(test_labels, torch.Tensor) else np.array(test_labels)
    
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    dataset='doore'
    k_fold=5
    delay_penalty=10
    padding_type='mean'
    augment=True
    aug_method='noise'

    fold_metrics = {
        "accuracy": [],
        "earliness": [],
        "cost": []
    }
    per_fold_results = []
    

    for fold_idx in range(k_fold):
        
        print(f"\n=== Fold {fold_idx} ===")
        fold_path = f'fold_data/{dataset}/fold_{fold_idx}.pkl'

        X_train, y_train, X_test, y_test = load_example_data(dataset='doore', fold_idx=0, padding_type='mean', augment=True, aug_method='noise')   
        print(X_train.shape)

        delay_penalty = 10
        model = CALIMERA(delay_penalty=delay_penalty)
        model.fit(X_train, y_train)

        stop_timestamps, y_pred = model.test(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        earliness = sum(stop_timestamps) / (X_test.shape[-1] * X_test.shape[0])
        cost = 1.0 - accuracy + delay_penalty * earliness
        print(f"Accuracy: {accuracy:.4f} | Earliness: {earliness:.4f} | Cost: {cost:.4f}")

        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["earliness"].append(earliness)
        fold_metrics["cost"].append(cost)
        per_fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy,
            "earliness": earliness,
            "cost": cost
        })

    # Print and save final summary
    print("\n=== K-Fold Summary ===")
    for key in fold_metrics:
        values = np.array(fold_metrics[key])
        print(f"{key.capitalize()}: {values.mean():.4f} ± {values.std():.4f}")

    save_kfold_summary_to_csv(
        dataset,
        metric_dict=fold_metrics,
        early_acc_by_step={},  # not used here
        early_acc_by_fold=per_fold_results,
        filename=f"results/calimera_{dataset}_summary.csv"
    )

