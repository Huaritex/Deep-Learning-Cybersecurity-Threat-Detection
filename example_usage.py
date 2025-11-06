"""
Example script to demonstrate model usage for threat detection
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class ThreatDetector(nn.Module):
    """Neural network for cybersecurity threat detection"""
    
    def __init__(self, input_features=7):
        super(ThreatDetector, self).__init__()
        self.fc1 = nn.Linear(input_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model(model_path='threat_detector.pth', input_features=7):
    """Load a trained model"""
    model = ThreatDetector(input_features=input_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_threat(model, data, scaler=None):
    """
    Predict if network events are malicious or benign
    
    Args:
        model: Trained ThreatDetector model
        data: DataFrame or numpy array with features
        scaler: Optional StandardScaler for preprocessing
    
    Returns:
        predictions: Binary predictions (0 or 1)
        probabilities: Probability scores
    """
    # Convert to tensor
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Scale if scaler provided
    if scaler:
        data = scaler.transform(data)
    
    # Convert to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        logits = model(data_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).int()
    
    return predictions.numpy(), probabilities.numpy()


def main():
    """Example usage"""
    # Example: Create sample data
    sample_data = pd.DataFrame({
        'processId': [30711, 10956, 75432],
        'threadId': [47, 783, 2341],
        'parentProcessId': [7867, 9489, 234],
        'userId': [1869, 1081, 42],
        'mountNamespace': [4, 7, 25],
        'argsNum': [0, 0, 15],
        'returnValue': [0, 1, -1]
    })
    
    print("Sample Network Events:")
    print(sample_data)
    print("\n" + "="*50)
    
    # Load model (you would need to train and save it first)
    # model = load_model('threat_detector.pth')
    
    # For demo purposes, create a new model
    model = ThreatDetector(input_features=7)
    
    # Make predictions
    predictions, probabilities = predict_threat(model, sample_data)
    
    print("\nPredictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "🚨 MALICIOUS" if pred == 1 else "✅ BENIGN"
        print(f"Event {i+1}: {status} (Confidence: {prob[0]*100:.2f}%)")


if __name__ == "__main__":
    main()
