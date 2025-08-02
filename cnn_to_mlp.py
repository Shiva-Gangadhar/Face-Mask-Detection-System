import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNtoMLP(nn.Module):
    def __init__(self, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate):
        super(CNNtoMLP, self).__init__()
        # CNN feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)   # Input channels=3, Output=32
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Output=64
        
        # Calculate flattened size after conv and pooling layers (assuming input size 128x128)
        self.flattened_size = 64 * 30 * 30
        
        # Build MLP layers dynamically based on params
        layers = []
        input_dim = self.flattened_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, neurons_per_layer))
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = neurons_per_layer
        
        # Final output layer
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # CNN forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.reshape(x.size(0), -1)
        # MLP forward pass
        return self.mlp(x)

# Example usage: saving and loading model weights
if __name__ == "__main__":
    # Create model instance with example parameters
    model = CNNtoMLP(output_dim=2, num_hidden_layers=4, neurons_per_layer=112, dropout_rate=0.1)
    
    # Save model weights
    torch.save(model.state_dict(), "model_weights.pth")
    
    # To load weights later:
    # model = CNNtoMLP(output_dim=2, num_hidden_layers=2, neurons_per_layer=64, dropout_rate=0.3)
    # model.load_state_dict(torch.load("model_weights.pth"))
    # model.eval()
