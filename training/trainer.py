import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import os

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, device='cuda', save_dir='save_model', save_name='best_model.pth'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_state = None

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        all_preds = []
        all_labels = []

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # --- Validation ---
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {np.mean(train_losses):.4f}, Train Acc: {train_acc:.4f} | Val Loss: {np.mean(val_losses):.4f}, Val Acc: {val_acc:.4f}")

        # Save best model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Best model updated and saved to {save_path}")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model  # Return best model
