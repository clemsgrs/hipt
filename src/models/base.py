import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x, **kwargs):
        raise NotImplementedError(
            "Forward method must be implemented in the subclass."
        )

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str


class LinearLayer(BaseModel):
    def __init__(self, input_dim: int, output_dim: int):
        """
        A simple linear layer for classification or regression tasks.
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features (e.g., number of classes).
        """
        super(LinearLayer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiLayerPerceptron(BaseModel):
    """
    A simple MLP with a tunable number of hidden layers.
    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of output features (e.g., number of classes).
        num_layers (int): Number of hidden layers in the MLP.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)