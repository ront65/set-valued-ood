from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Tuple

class Metric(ABC):
    name='a'
    name = None

    @staticmethod
    @abstractmethod
    def calc(model_preds : NDArray[Tuple[int, int]], y_true : NDArray[int], domains : NDArray[int]):
        pass

    @classmethod
    @abstractmethod
    def plot(cls, model_preds : NDArray[Tuple[int, int]], y_true : NDArray[int], domains : NDArray[int], ax, label, **kwargs):
        pass

