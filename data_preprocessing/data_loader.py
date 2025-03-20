import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter

from data_preprocessing.data_preprocess import preprocess_dataset, pad_sequences, balance_by_augmentation
from data_preprocessing.doore_preprocess import dooreLoader
from data_preprocessing.opportunity_preprocess import opportunityLoader
from data_preprocessing.casas_preprocess import casasLoader
from data_preprocessing.aras_preprocess import arasLoader
from data_preprocessing.openpack_preprocess import openpackLoader

import logging

def load_and_preprocess_data(args, mode='train'):
    # 1. Load dataset
    if args.dataset == 'doore':
        dataset_list = dooreLoader('data/doore/*.csv', timespan=args.timespan, min_seq=args.min_seq)
    elif args.dataset == 'opportunity':
        dataset_list = opportunityLoader('data/opportunity/*.dat', timespan=args.timespan, min_seq=args.min_seq)
    elif args.dataset == 'casas':
        dataset_list = casasLoader('data/casas/*.txt', min_seq=args.min_seq)
    elif args.dataset == 'aras':
        dataset_list = arasLoader('data/aras/HouseA/*.txt', timespan=args.timespan, min_seq=args.min_seq)
    elif args.dataset == 'openpack':
        dataset_list = openpackLoader('data/openpack/*.csv', timespan=args.timespan, min_seq=args.min_seq)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 2. Filter by class count
    if args.min_samples > 0:
        raw_labels = [seq.label for seq in dataset_list]
        counts = Counter(raw_labels)
        valid_labels = {label for label, count in counts.items() if count >= args.min_samples}
        dataset_list = [seq for seq in dataset_list if seq.label in valid_labels]
        logging.info(f"[Filter] Classes ≥ {args.min_samples}: {sorted(valid_labels)}")

        # Re-count filtered data
        filtered_counts = Counter(seq.label for seq in dataset_list)
        total_sequences_filtered = sum(filtered_counts.values())
        total_pointers_filtered = sum(seq.length for seq in dataset_list)
        logging.info(f"[After Filter] Total sequences: {total_sequences_filtered}")
        logging.info(f"[After Filter] Total data points: {total_pointers_filtered}")
        for label in sorted(filtered_counts.keys()):
            count = filtered_counts[label]
            total_points = sum(seq.length for seq in dataset_list if seq.label == label)
            logging.info(f"  Activity {label}: {count} sequences, {total_points} data points")

    # 3. Normalize + Relabel (No augment, No padding)
    normalized_seqs, label_list, label_map = preprocess_dataset(
        dataset_list,
        padding_type=None,
        augment_method=None
    )

    labels_tensor = torch.tensor(label_list, dtype=torch.long)

    # 4. Split into train/val/test
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        normalized_seqs, labels_tensor, test_size=args.test_ratio, stratify=labels_tensor, random_state=42
    )
    val_ratio_adjusted = args.valid_ratio / (1 - args.test_ratio)
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=val_ratio_adjusted, stratify=train_val_labels, random_state=42
    )

    # 5. Augment train only (if enabled)
    if mode == 'train' and args.augment:
        logging.info(f"[Augment] Method: {args.aug_method}")
        train_data, train_labels = balance_by_augmentation(train_data, train_labels, method=args.aug_method)

    # 6. Padding all sets
    logging.info("[Padding] Applying padding...")
    train_tensor, train_lengths = pad_sequences(train_data, padding_type=args.padding)
    val_tensor, val_lengths = pad_sequences(val_data, padding_type=args.padding)
    test_tensor, test_lengths = pad_sequences(test_data, padding_type=args.padding)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(train_tensor, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_labels), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_tensor, test_labels), batch_size=args.batch_size)

    logging.info(f"  Train Tensor: {train_tensor.shape}")
    logging.info(f"  Val Tensor:   {val_tensor.shape}")
    logging.info(f"  Test Tensor:  {test_tensor.shape}")
    logging.info(f"[Count] Train: {len(train_tensor)}, Val: {len(val_tensor)}, Test: {len(test_tensor)}")

    # Class-wise stats
    # print("\n[Train Set] Class-wise stats:")
    # _print_class_stats(train_labels, train_lengths)

    # print("\n[Val Set] Class-wise stats:")
    # _print_class_stats(val_labels, val_lengths)

    # print("\n[Test Set] Class-wise stats:")
    # _print_class_stats(test_labels, test_lengths)

    return train_loader, val_loader, test_loader, label_map, train_tensor.shape[2]  # num_channels

def _print_class_stats(labels_tensor, length_list):
    label_counts = Counter(labels_tensor.tolist())
    for label in sorted(label_counts.keys()):
        seq_count = label_counts[label]
        total_points = sum(length for lbl, length in zip(labels_tensor.tolist(), length_list) if lbl == label)
        print(f"  Class {label}: {seq_count} sequences, {total_points} data points")
