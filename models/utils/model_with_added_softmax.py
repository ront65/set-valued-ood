import torch


class ModelWithAddedSoftmax(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(ModelWithAddedSoftmax, self).__init__()

        self._model = model 
        self.softmax = torch.nn.Softmax(dim=1)

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        model_outputs = self._model(X)
        return self.softmax(model_outputs)