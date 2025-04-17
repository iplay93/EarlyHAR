import torch
import argparse
import os
import yaml
import logging
import wandb
import pickle

from data_preprocessing.data_loader import prepare_kfold_datasets, load_fold_data
from models.deepcov_lstm import DeepConvLSTM
from models.transformer import TransformerClassifier
from models.mlstm import MLSTM_FCN  
from training.trainer import train_model
from evals.evaluation import evaluate_model, evaluate_early_classification
from utils.logger import setup_logging
from utils.wandb import manage_wandb_logs

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
    parser.add_argument('--model_type', type=str, default='deepconvlstm', choices=['deepconvlstm', 'transformer', 'mlstm'])
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
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


def main():
    torch.cuda.empty_cache()
    args = setup_args()

    # --- Logging Setup ---
    manage_wandb_logs(wandb_dir='wandb', max_runs=5)
    log_path = setup_logging(base_log_dir='logs', dataset_name=args.dataset, mode=args.mode, max_logs=5)
    logging.info(f"Using device: {args.device}")

    # --- Data Preparation (run only once if needed) ---
    if args.prepare_data or not os.path.exists(f"fold_data/{args.dataset}/fold_0.pkl"):
        label_map, input_channels = prepare_kfold_datasets(args)
    else:
        with open(f"fold_data/{args.dataset}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        label_map = meta["label_map"]
        input_channels = meta["input_channels"]

    # --- K-Fold Training/Evaluation Loop ---
    for fold_id in range(args.k_fold):
        logging.info(f"\n=== Loading Fold {fold_id} ===")
        fold_path = f"fold_data/{args.dataset}/fold_{fold_id}.pkl"
        train_loader, val_loader, test_loader = load_fold_data(fold_path, args, mode='train')

        # --- Model Initialization ---
        model = setup_model(args, input_channels, num_classes=len(label_map))

        # --- Wandb Run per Fold ---
        wandb.init(project="EarlyHAR", name=f"{args.dataset}_fold{fold_id}", config=vars(args))
        wandb.watch(model, log='all', log_freq=100)

        if args.mode == 'train':
            logging.info("Starting training...")
            train_model(model, train_loader, val_loader, args)
            
            # --- Save Model ---
            ckpt_dir = os.path.join("checkpoints", args.dataset)
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"fold{fold_id}.pt"))

        elif args.mode == 'test':
            logging.info("Starting evaluation...")
            model.load_state_dict(torch.load(f"checkpoints/{args.dataset}_fold{fold_id}.pt"))
            results = evaluate_model(model, test_loader, args)

            wandb.log({
                "Test Accuracy": results['accuracy'],
                "Test Precision": results['precision'],
                "Test Recall": results['recall'],
                "Test F1": results['f1_score']
            })

            # --- Early Classification Evaluation ---
            early_results = evaluate_early_classification(model, test_loader, args)
            for step, acc in early_results.items():
                logging.info(f"Early Accuracy at {int(step * 100)}% sequence: {acc:.4f}")

        # --- Cleanup ---
        wandb.finish()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
