import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

class ConformalClassifier:
    def __init__(
        self, 
        base_classifier, 
        num_classes: int, 
        pretraiend: bool = False, 
        coverage: float = 0.9, 
        minimal_cnt_for_calibration: int = 1
    ):
        self.base_classifier = base_classifier
        self.pretraiend = pretraiend
        self.num_classes = num_classes
        self.coverage = coverage
        self.minimal_cnt_for_calibration = minimal_cnt_for_calibration


    def fit(self, X: ArrayLike, y: ArrayLike):
        if not self.pretraiend:
            self.base_classifier.fit(X, y)
        return self


    def calibrate_from_preds(self, preds: np.ndarray, y: np.ndarray) -> None:
        assert preds.ndim == 2 and y.ndim == 1
        assert preds.shape[0] == y.shape[0]
        assert preds.shape[1] == self.num_classes

        self.scores: dict[int, np.ndarray] = None
        self.scores = {
            i: 1 - (preds[y == i, i])
            for i in range(self.num_classes)
        }

        self.qhats: np.ndarray = None
        self.qhats = np.array([
            (np.quantile(self.scores[i], self.coverage) if len(self.scores[i]) >= self.minimal_cnt_for_calibration else np.nan)
            for i in range(self.num_classes)
        ])
        assert self.qhats.ndim == 1 and self.qhats.shape[0] == self.num_classes


    def calibrate(self, X: ArrayLike, y: np.ndarray) -> None:
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        preds = self.base_classifier.predict_proba(X) # Preds should be np.array
        self.calibrate_from_preds(preds, y)       


    def predict(self, X: ArrayLike, base_model_logits: Optional[np.ndarray] = None) -> np.ndarray:
        assert X is None or base_model_logits is None
        assert not (X is None and base_model_logits is None)

        if X is not None:
            base_model_logits = self.base_classifier.predict_proba(X) #Expectes predict_proba tp return np.array

        assert isinstance(base_model_logits, np.ndarray)
        assert base_model_logits.ndim == 2
        assert base_model_logits.shape[1] == self.num_classes
        
        return base_model_logits >= 1 - self.qhats
        



