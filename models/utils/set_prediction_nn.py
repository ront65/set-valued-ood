import torch
from typing import Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class SetPredictionNN(torch.nn.Module, ABC):

    def __init__(self, model: torch.nn.Module):
        super(SetPredictionNN, self).__init__()
        self._model = model 

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._model(X)
    

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        
        res = self.forward(X)
        return res.detach().numpy()
                

    def predict_classes(
            self, 
            X: torch.Tensor , 
            return_logits: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        model_outputs = self(X)
        classes = self._get_class_from_model_output(model_outputs)
        
        assert classes.ndim == model_outputs.ndim == 2
        assert classes.shape[0] == model_outputs.shape[0]
        assert classes.shape[1] == model_outputs.shape[1]
        
        if return_logits:
            return classes, model_outputs
        else:
            return classes
        
    @abstractmethod
    def _get_class_from_model_output(self, model_output: torch.Tensor) -> torch.Tensor:
        pass