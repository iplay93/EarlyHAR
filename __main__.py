import torch
import argparse
import os
import yaml
import logging
import wandb

from data_preprocessing.data_loader import load_and_preprocess_data
from models.deepcov_lstm import DeepConvLSTM
from models.transformer import TransformerClassifier
from training.trainer import train_model
from evals.evaluation import evaluate_model
from utils.logger import setup_logging


# ---------------- Model Setup ----------------
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
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

# ---------------- Sweep Function ----------------
def sweep_train():
    wandb.init()
    config = wandb.config

    # Extract config into args
    args = argparse.Namespace(
        dataset=config.dataset,
        mode='train',
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        train_epochs=config.train_epochs,
        model_type=config.model_type,
        padding=config.padding,
        augment=config.augment,
        aug_method=config.aug_method,
        early_stop=config.early_stop,
        patience=config.patience,
        save_dir='save_model',
        save_name=f"{config.dataset}_{config.model_type}_sweep.pth",
        save_path=os.path.join('save_model', f"{config.dataset}_{config.model_type}_sweep.pth"),
        cm_image_path=f"results/{config.dataset}_{config.model_type}_cm.png",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        pooling=config.pooling,
        use_cls_token=config.use_cls_token,
        use_batchnorm=config.use_batchnorm,
        classifier_hidden=config.classifier_hidden,
        test_ratio=config.test_ratio,
        valid_ratio=config.valid_ratio,
        timespan=config.timespan,
        min_seq=config.min_seq,
        min_samples=config.min_samples
    )

    log_path = setup_logging('logs', dataset_name=args.dataset, mode='sweep', max_logs=5)
    train_loader, val_loader, _, label_map, input_channels = load_and_preprocess_data(args, mode='train')
    model = setup_model(args, input_channels, len(label_map))
    wandb.watch(model, log='all', log_freq=100)
    train_model(model, train_loader, val_loader, args)
    wandb.finish()

