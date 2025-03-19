import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.model_selection import train_test_split

from data_preprocessing.doore_processing import dooreLoader
from data_preprocessing.data_loader import preprocess_dataset
from models.deepcov_lstm import DeepConvLSTM
from training.trainer import train_model
from evals.evaluation import evaluate_model

# Argument parsing separated into a function
def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='doore', help='Dataset name')
    parser.add_argument('--timespan', type=int, default=10000, help='Timespan between data points (ms)')
    parser.add_argument('--min_seq', type=int, default=10, help='Minimum sequence length')
    parser.add_argument('--padding', type=str, default='mean', help='Padding type: max or mean')
    parser.add_argument('--augment', type=bool, default=True, help='Use data augmentation')
    parser.add_argument('--aug_method', type=str, default='noise', help='Augmentation method: noise, permute, scaling')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Validation set ratio')

    args = parser.parse_args()
    return args


def main():
    args = setup_args()

    # Load raw dataset (currently only supports doore)
    if args.dataset == 'doore':
        dataset_list = dooreLoader('data/doore/*.csv', timespan=args.timespan, min_seq=args.min_seq)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Preprocess: normalize, pad, relabel, augment
    data_tensor, label_list, label_map = preprocess_dataset(
        dataset_list,
        padding_type=args.padding,
        augment_method=args.aug_method if args.augment else None
    )

    print("Data tensor shape:", data_tensor.shape)
    print("Label distribution:", Counter(label_list))

    # Split into train/val/test
    labels_tensor = torch.tensor(label_list, dtype=torch.long)
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        data_tensor, labels_tensor, test_size=args.test_ratio, stratify=labels_tensor, random_state=42
    )
    val_ratio_adjusted = args.valid_ratio / (1 - args.test_ratio)
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=val_ratio_adjusted, stratify=train_val_labels, random_state=42
    )

    # DataLoaders
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=args.batch_size)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepConvLSTM(input_channels=data_tensor.shape[2], num_classes=len(label_map))

    # Train
    # trained_model = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     num_epochs=args.train_epochs,
    #     lr=args.learning_rate,
    #     device=device,
    #     save_dir='save_model',
    #     save_name='doore_deepconvlstm.pth'
    #     )


    # Assume model architecture is defined
    model = DeepConvLSTM(input_channels=data_tensor.shape[2], num_classes=len(label_map))

    # Evaluate saved model
    results = evaluate_model(
        model,
        test_loader,
        save_path='save_model/doore_deepconvlstm.pth',
        device=device,
        cm_image_path='results/doore_normalized_cm.png'
    )

    print("Returned metrics:", results)

if __name__ == '__main__':
    main()
