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