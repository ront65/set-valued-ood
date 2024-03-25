import torch
from models.utils.set_prediction_nn import SetPredictionNN



class MaxOutputAsClassPred(SetPredictionNN):

    def _get_class_from_model_output(self, model_output: torch.Tensor) -> torch.Tensor:
        assert model_output.dim() == 2 
        
        _, max_indices = torch.max(model_output, dim=1)

        # Create a tensor of 0-1 with a single 1 per row for the highest output
        one_hot_tensor = (
            torch.zeros_like(model_output)
            .scatter_(
                dim=1, 
                index=max_indices.unsqueeze(1), 
                value=1.0
            )
        )
        return one_hot_tensor

