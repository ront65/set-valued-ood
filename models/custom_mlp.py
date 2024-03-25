import torch

class CustomMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2):
        super(CustomMLP, self).__init__()
        hidden_dim = int(input_dim/2)
        # Define the layers
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim, dtype=torch.float32)
        
        # Activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.output_layer(x)  # Output layer usually doesn't have activation
        return x