import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import logging

import wandb


def train_model(model, train_loader, val_loader, args, return_val_score=False):
    """
    Trains a model with optional early stopping.
    If val_loader is None, performs full training without validation.
    If return_val_score is True, returns final validation accuracy for model selection.
    """
    device = args.device
    num_epochs = args.train_epochs
    learning_rate = args.learning_rate
    early_stop = args.early_stop
    patience = args.patience
    save_path = args.save_path

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # === Validation ===
        if val_loader is not None:
            model.eval()
            val_losses, val_preds, val_labels = [], [], []

            with torch.no_grad():
                for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            val_acc = accuracy_score(val_labels, val_preds)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs} | "
                             f"Train Loss: {np.mean(train_losses):.4f}, Train Acc: {train_acc:.4f} | "
                             f"Val Loss: {np.mean(val_losses):.4f}, Val Acc: {val_acc:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": np.mean(train_losses),
                "train_acc": train_acc,
                "val_loss": np.mean(val_losses),
                "val_acc": val_acc
            })

            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                torch.save(best_model_state, save_path)
                logging.info(f"Best model updated and saved to {save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if early_stop and epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break
        else:
            # No validation (full training) → skip early stopping
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs} | "
                             f"Train Loss: {np.mean(train_losses):.4f}, Train Acc: {train_acc:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": np.mean(train_losses),
                "train_acc": train_acc,
            })

    if best_model_state:
        model.load_state_dict(best_model_state)

    if return_val_score:
        return best_val_acc
    return model