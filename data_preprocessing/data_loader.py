import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from collections import Counter
import math
import random

# 1. Z-score normalization
def z_score_normalize(data_array):
    df = pd.DataFrame(data_array)
    df_norm = (df - df.mean()) / df.std()
    df_norm = df_norm.fillna(0)
    return df_norm.to_numpy()

# 2. Relabel to 0-based continuous labels
def relabel_continuous(label_list):
    unique_labels = sorted(set(label_list))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    relabeled = [label_map[label] for label in label_list]
    return relabeled, label_map

# 3. Pad sequences to fixed length (max or mean)
def pad_sequences(sequence_list, padding_type='max'):
    lengths = [len(seq) for seq in sequence_list]
    max_len = max(lengths)
    mean_len = int(np.mean(lengths))

    pad_len = max_len if padding_type == 'max' else mean_len
    padded_list = []

    for seq in sequence_list:
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        if len(seq) < pad_len:
            pad_size = (0, 0, 0, pad_len - len(seq))  # Pad time dimension
            padded_seq = F.pad(seq_tensor, pad_size, "constant", 0)
        else:
            padded_seq = seq_tensor[:pad_len]
        padded_list.append(padded_seq)

    data_tensor = torch.stack(padded_list)
    return data_tensor, lengths

# 4. Augmentation methods
def add_noise(tensor, noise_std=0.01):
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise

def time_scaling(tensor, scale_factor=1.2):
    length = tensor.shape[0]
    new_length = int(length * scale_factor)
    return F.interpolate(tensor.unsqueeze(0).transpose(1,2), size=new_length, mode='linear', align_corners=False).transpose(1,2).squeeze(0)

def permute_segments(tensor, num_segments=4):
    length = tensor.shape[0]
    segment_size = length // num_segments
    segments = [tensor[i*segment_size:(i+1)*segment_size] for i in range(num_segments)]
    random.shuffle(segments)
    return torch.cat(segments, dim=0)

# 5. Augmentation balancing with dynamic padding
def balance_by_augmentation(normalized_seqs, label_list, method='noise', target_count=None):
    label_counts = Counter(label_list)
    max_count = target_count or max(label_counts.values())

    augmented_data = []
    augmented_labels = []

    for label in sorted(set(label_list)):
        idxs = [i for i, lbl in enumerate(label_list) if lbl == label]
        samples_needed = max_count - len(idxs)

        # Add original sequences
        for idx in idxs:
            tensor_seq = torch.tensor(normalized_seqs[idx], dtype=torch.float32)
            augmented_data.append(tensor_seq)
            augmented_labels.append(label)

        # Generate augmented sequences
        if samples_needed > 0:
            for _ in range(samples_needed):
                source_idx = random.choice(idxs)
                original = torch.tensor(normalized_seqs[source_idx], dtype=torch.float32)

                if method == 'noise':
                    aug = add_noise(original)
                elif method == 'scaling':
                    aug_scaled = time_scaling(original)
                    aug = F.interpolate(aug_scaled.unsqueeze(0).transpose(1,2),
                                        size=original.shape[0],
                                        mode='linear',
                                        align_corners=False).transpose(1,2).squeeze(0)
                elif method == 'permute':
                    aug = permute_segments(original)
                    # Padding/cropping for equal length
                    if aug.shape[0] != original.shape[0]:
                        if aug.shape[0] > original.shape[0]:
                            aug = aug[:original.shape[0]]
                        else:
                            pad_size = (0, 0, 0, original.shape[0] - aug.shape[0])
                            aug = F.pad(aug, pad_size, "constant", 0)
                else:
                    aug = original  # No augmentation

                augmented_data.append(aug)
                augmented_labels.append(label)

    return augmented_data, augmented_labels

# 6. Full preprocessing pipeline
def preprocess_dataset(dataset_list, padding_type='max', augment_method=None):
    print("Starting preprocessing pipeline...")

    # Extract raw sequences and labels
    raw_data = [seq.data for seq in dataset_list]
    labels = [seq.label for seq in dataset_list]

    # Flatten for normalization
    flattened = np.concatenate(raw_data, axis=0)
    normalized = z_score_normalize(flattened)

    # Split normalized data into sequences
    normalized_seqs = []
    idx = 0
    for seq in raw_data:
        seq_len = len(seq)
        norm_seq = normalized[idx:idx + seq_len]
        normalized_seqs.append(norm_seq)
        idx += seq_len

    # Relabel activities to 0-based
    relabeled, label_map = relabel_continuous(labels)

    # Optional Data Augmentation BEFORE padding
    if augment_method is not None:
        augmented_seqs, relabeled = balance_by_augmentation(normalized_seqs, relabeled, method=augment_method)
    else:
        augmented_seqs = [torch.tensor(seq, dtype=torch.float32) for seq in normalized_seqs]

    # Padding
    padded_tensor, _ = pad_sequences(augmented_seqs, padding_type)

    print(f"Final dataset size: {padded_tensor.shape[0]} samples, each of shape {padded_tensor.shape[1:]}")
    return padded_tensor, relabeled, label_map

from doore_processing import dooreLoader

# Load dataset (as TSDataSet list)
dataset_list = dooreLoader('../data/doore/*.csv', timespan=10000, min_seq=10)

# Preprocess with 'permute' augmentation and 'mean' padding
data_tensor, label_list, label_map = preprocess_dataset(dataset_list, padding_type='mean', augment_method='noise')

# Output check
print("Data tensor shape:", data_tensor.shape)
print("Label counts:", Counter(label_list))
print("Label mapping:", label_map)