import torch
from typing import Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike
from models.utils.max_outout_as_class_pred import MaxOutputAsClassPred


class DomainBedWrapper(MaxOutputAsClassPred):

    def __init__(self, model: torch.nn.Module):
        super(DomainBedWrapper, self).__init__(model)

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._model.predict(X)
    
    def update(self, minibatches):
        return self._model.update(minibatches)['loss']
    
        
