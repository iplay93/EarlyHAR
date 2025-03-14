import torch
import torch.nn as nn
import torch.nn.functional as F

class CoT_EarlyTSC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, threshold=0.9):
        """
        Chain of Thought-based Early Time Series Classification model.
        
        Args:
        - input_dim (int): The number of input features per time step.
        - hidden_dim (int): LSTM hidden state dimension.
        - num_classes (int): The number of classification categories.
        - threshold (float): Confidence threshold for early classification.
        """
        super(CoT_EarlyTSC, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.threshold = threshold  # If confidence >= threshold, classify early
    
    def forward(self, x, step=5):
        """
        Performs early classification using only the first `step` time steps.
        
        Args:
        - x (Tensor): Input time series of shape (batch, seq_len, input_dim).
        - step (int): The number of initial time steps to consider.
        
        Returns:
        - preds (Tensor): Predicted class indices.
        - confidence (Tensor): Confidence scores for predictions.
        - early_classification (Tensor): Boolean mask indicating early classification.
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)

        # Chain of Thought - Early time series reasoning (use only first `step` time steps)
        x_partial = x[:, :step, :]  # Use only the first `step` time steps
        out, _ = self.lstm(x_partial, (h, c))
        out = out[:, -1, :]  # Take the last output of the LSTM
        logits = self.fc(out)  # Convert to class logits
        probs = F.softmax(logits, dim=1)  # Convert to probability distribution

        # Early classification decision (if max probability >= threshold, classify early)
        confidence, preds = torch.max(probs, dim=1)
        early_classification = confidence >= self.threshold

        return preds, confidence, early_classification


# Initialize model
input_dim = 3   # Example input feature size (e.g., 3 sensors)
hidden_dim = 32
num_classes = 5
threshold = 0.85  # Classify early if confidence is ≥ 85%

model = CoT_EarlyTSC(input_dim, hidden_dim, num_classes, threshold)

# Create dummy time series data (batch_size=4, seq_len=20, input_dim=3)
x = torch.rand(4, 20, input_dim)

# Run inference using only the first 5 time steps
preds, confidence, early_classification = model(x, step=5)

print("Predicted classes:", preds)
print("Confidence scores:", confidence)
print("Early classification decision:", early_classification)
