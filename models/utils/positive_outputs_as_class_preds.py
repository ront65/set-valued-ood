import torch
from models.utils.set_prediction_nn import SetPredictionNN

class PositiveOutputsAsClassPreds(SetPredictionNN):

    def _get_class_from_model_output(self, model_output: torch.Tensor) -> torch.Tensor:
        assert model_output.dim() == 2 
        return (model_output > 0).float()
