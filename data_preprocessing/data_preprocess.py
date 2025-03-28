import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import Counter
import random

# --------- Normalization ---------
def z_score_normalize(sequence_list):
    # Concatenate all sequences vertically (time axis)
    flat_data = np.concatenate(sequence_list, axis=0)  # Shape: [total_length, channels]
    df = pd.DataFrame(flat_data)

    # Prevent division by zero (std=0) → replace with 1 temporarily
    std_replaced = df.std().replace(0, 1)
    df_norm = (df - df.mean()) / std_replaced

    normalized_flat = df_norm.to_numpy()

    # Reconstruct per sequence
    normalized_sequences = []
    idx = 0
    for seq in sequence_list:
        length = len(seq)
        norm_seq = normalized_flat[idx:idx + length]
        normalized_sequences.append(norm_seq)
        idx += length

    return normalized_sequences

# --------- Padding ---------
def pad_sequences(sequence_list, padding_type='mean'):
    lengths = [len(seq) for seq in sequence_list]
    max_len = max(lengths)
    mean_len = int(np.mean(lengths))
    pad_len = max_len if padding_type == 'max' else mean_len

    padded_tensors = []
    for seq in sequence_list:
        tensor_seq = torch.tensor(seq, dtype=torch.float32)
        if len(seq) < pad_len:
            pad_size = (0, 0, 0, pad_len - len(seq))
            padded_seq = F.pad(tensor_seq, pad_size, mode='constant', value=0)
        else:
            padded_seq = tensor_seq[:pad_len]
        padded_tensors.append(padded_seq)

    data_tensor = torch.stack(padded_tensors)
    return data_tensor, lengths

# --------- Relabeling ---------
def relabel_continuous(label_list):
    unique_labels = sorted(set(label_list))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    relabeled = [label_map[label] for label in label_list]
    return relabeled, label_map

# --------- Augmentation Methods ---------
def add_noise(tensor, std=0.01):
    noise = torch.randn_like(tensor) * std
    return tensor + noise

def time_scaling(tensor, scale_factor=1.2):
    length = tensor.shape[0]
    new_len = int(length * scale_factor)
    scaled = F.interpolate(tensor.unsqueeze(0).transpose(1,2), size=new_len, mode='linear', align_corners=False)
    scaled = scaled.transpose(1,2).squeeze(0)
    return scaled

def permute_segments(tensor, num_segments=4):
    length = tensor.shape[0]
    seg_size = length // num_segments
    segments = [tensor[i*seg_size:(i+1)*seg_size] for i in range(num_segments)]
    random.shuffle(segments)
    return torch.cat(segments, dim=0)

# --------- Augmentation Wrapper ---------
def balance_by_augmentation(sequence_list, label_list, method='noise', target_count=None):
    label_counts = Counter(label_list)
    max_count = target_count or max(label_counts.values())

    augmented_data = []
    augmented_labels = []

    for label in sorted(set(label_list)):
        idxs = [i for i, lbl in enumerate(label_list) if lbl == label]
        samples_needed = max_count - len(idxs)

        # Original sequences
        for idx in idxs:
            tensor_seq = torch.tensor(sequence_list[idx], dtype=torch.float32)
            augmented_data.append(tensor_seq)
            augmented_labels.append(label)

        # Augmented samples
        for _ in range(samples_needed):
            source_idx = random.choice(idxs)
            original = torch.tensor(sequence_list[source_idx], dtype=torch.float32)

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
                if aug.shape[0] != original.shape[0]:
                    aug = aug[:original.shape[0]] if aug.shape[0] > original.shape[0] else F.pad(aug, (0,0,0,original.shape[0]-aug.shape[0]), mode='constant', value=0)
            else:
                aug = original  # No augmentation

            augmented_data.append(aug)
            augmented_labels.append(label)

    return augmented_data, torch.tensor(augmented_labels, dtype=torch.long)

# --------- Unified Preprocessing ---------
def preprocess_dataset(dataset_list, padding_type='mean', augment_method=None):
    raw_sequences = [seq.data for seq in dataset_list]
    labels = [seq.label for seq in dataset_list]

    # Normalize
    normalized_seqs = z_score_normalize(raw_sequences)

    # Relabel to 0-based
    relabeled, label_map = relabel_continuous(labels)

    if augment_method is not None:
        normalized_seqs, relabeled = balance_by_augmentation(normalized_seqs, relabeled, method=augment_method)

    # Padding
    if padding_type is None:
        return normalized_seqs, relabeled, label_map
    else:   
        padded_tensor, lengths = pad_sequences(normalized_seqs, padding_type)
        return padded_tensor, relabeled, label_map
