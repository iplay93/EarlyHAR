import torch
import argparse
import os
import yaml
import logging
import wandb
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing.data_loader import prepare_kfold_datasets, load_fold_data
from models.deepcov_lstm import DeepConvLSTM
from models.transformer import TransformerClassifier
from models.mlstm import MLSTM_FCN  
from training.trainer import train_model
from evals.evaluation import evaluate_model, evaluate_early_classification
from utils.logger import setup_logging
from utils.wandb import manage_wandb_logs
from utils.summary import print_kfold_summary, save_kfold_summary_to_csv
from data_preprocessing.data_preprocess import preprocess_dataset, pad_sequences, balance_by_augmentation

def load_yaml_config(dataset_name):
    config_path = os.path.join('configs', f'{dataset_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_args():
    parser = argparse.ArgumentParser()
    # --- General Arguments ---
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='doore')
    parser.add_argument('--padding', type=str, default='mean')
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--aug_method', type=str, default='noise')
    


    # --- Model Arguments ---
    parser.add_argument('--model_type', type=str, default='transformer', choices=['deepconvlstm', 'transformer', 'mlstm'])
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ff_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'cls'])
    parser.add_argument('--use_cls_token', type=bool, default=False)
    parser.add_argument('--use_batchnorm', type=bool, default=True)
    parser.add_argument('--classifier_hidden', type=int, default=64)

    # --- Training Arguments ---
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--prepare_data', action='store_true', help='Whether to generate and save k-fold data')

    # --- Parse and Merge ---
    args = parser.parse_args()
    dataset_config = load_yaml_config(args.dataset)
    for key, value in dataset_config.items():
        setattr(args, key, value)

    # --- Paths ---
    args.save_dir = getattr(args, 'save_dir', 'save_model')
    os.makedirs(args.save_dir, exist_ok=True)

    args.save_name = getattr(args, 'save_name', f"{args.dataset}_{args.model_type}.pth")
    args.save_path = os.path.join(args.save_dir, args.save_name)

    args.cm_image_path = getattr(args, 'cm_image_path', f"results/{args.dataset}_{args.model_type}_cm.png")
    os.makedirs(os.path.dirname(args.cm_image_path), exist_ok=True)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def setup_model(args, input_channels, num_classes):
    if args.model_type == 'deepconvlstm':
        return DeepConvLSTM(input_channels=input_channels, num_classes=num_classes)
    elif args.model_type == 'transformer':
        return TransformerClassifier(
            input_dim=input_channels,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            pooling=args.pooling,
            use_cls_token=args.use_cls_token,
            use_batchnorm=args.use_batchnorm,
            classifier_hidden=args.classifier_hidden
        )
    elif args.model_type == 'mlstm':        
        return MLSTM_FCN(input_channels=input_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
from data_preprocessing.opportunity_preprocess import opportunityLoader
from collections import Counter

def main():
    torch.cuda.empty_cache()
    args = setup_args()
    manage_wandb_logs(wandb_dir='wandb', max_runs=5)

    

    dataset_list = opportunityLoader('data/opportunity/*.dat', timespan=args.timespan, min_seq=args.min_seq)

    # Filter classes with enough samples
    if args.min_samples > 0:
        raw_labels = [seq.label for seq in dataset_list]
        counts = Counter(raw_labels)
        valid_labels = {label for label, count in counts.items() if count >= args.min_samples}
        dataset_list = [seq for seq in dataset_list if seq.label in valid_labels]

        logging.info(f"[Filter] Classes ≥ {args.min_samples}: {sorted(valid_labels)}")

    # Normalize & Relabel
    normalized_seqs, label_list, label_map = preprocess_dataset(dataset_list, padding_type='mean', normalize=False)

    data_tensor, lengths = pad_sequences(normalized_seqs, padding_type=args.padding)
    labels_tensor = torch.tensor(label_list, dtype=torch.long)

    # Train/Val/Test Split
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_tensor, labels_tensor,
        test_size=args.test_ratio,
        stratify=labels_tensor,
        random_state=42
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels,
        test_size=args.valid_ratio / (1 - args.test_ratio),
        stratify=train_labels,
        random_state=42
    )

    # Augment
    if args.mode == 'train' and args.augment:
        logging.info(f"[Augment] Method: {args.aug_method}")
        train_data, train_labels = balance_by_augmentation(train_data, train_labels, method=args.aug_method)

    # Padding (reapplied if needed)
    train_tensor, train_lengths = pad_sequences(train_data, padding_type=args.padding)
    val_tensor, val_lengths = pad_sequences(val_data, padding_type=args.padding)
    test_tensor, test_lengths = pad_sequences(test_data, padding_type=args.padding)

    print(f"Train: {train_tensor.shape}, Val: {val_tensor.shape}, Test: {test_tensor.shape}")
    # Dataloaders
    train_loader = DataLoader(TensorDataset(train_tensor, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_labels), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_tensor, test_labels), batch_size=args.batch_size)

    logging.info(f"[Data Loaded] Train: {len(train_tensor)}, Val: {len(val_tensor)}, Test: {len(test_tensor)}")
    
    # Model
    model = setup_model(args, input_channels=train_tensor.shape[2], num_classes=len(label_map))

    wandb.init(project="EarlyHAR", name=f"{args.dataset}_fold{1}", config=vars(args))
    wandb.watch(model, log='all', log_freq=100)

    train_model(model, train_loader, val_loader, args)

    wandb.finish()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
