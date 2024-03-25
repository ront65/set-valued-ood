import pandas as pd
import numpy as np
from models.utils import ConformalClassifier
from numpy.typing import ArrayLike
from typing import Hashable, Optional

class RobustConformal:
    def __init__(
            self,
            base_moedl,
            num_classes: int,
            conformal_model_kwargs: dict = {},
            pretrained: bool = False, 
            coverage=0.9,
            minimal_cnt_for_calibration: int = 1,
    ):
        self.base_moedl = base_moedl
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.conformal_model_class = ConformalClassifier
        self.conformal_model_kwargs = conformal_model_kwargs
        self.conformal_models = None

        self.coverage = coverage
        self.minimal_cnt_for_calibration = minimal_cnt_for_calibration

    def fit(self, X: ArrayLike, y: ArrayLike, domains: ArrayLike):
        if not self.pretrained:
            self.base_moedl.fit(X, y)
    

    def calibrate(self, X: ArrayLike, y: np.ndarray, domains: np.ndarray) -> None:
        assert y.ndim == domains.ndim == 1
        assert X.shape[0] == y.shape[0] == domains.shape[0]

        preds = self.base_moedl.predict_proba(X) # Preds should be np.array
        self.calirate_from_preds(preds, y, domains)


    def calibrate_from_preds(self, preds: np.ndarray, y: np.ndarray, domains: np.ndarray) -> None:
        assert preds.ndim == 2 and y.ndim == 1 and domains.ndim == 1
        assert preds.shape[0] == y.shape[0] == domains.shape[0]
        assert preds.shape[1] == self.num_classes

        self.conformal_models = dict()

        unique_domains = np.unique(domains)
        for domain in unique_domains:
            self._calibrate_icp_single_domain(
                preds=preds[domains == domain],
                y=y[domains == domain],
                domain=domain
            )
    

    def predict(self, X: ArrayLike, base_model_logits: Optional[np.ndarray] = None) -> np.ndarray:
        assert X is None or base_model_logits is None
        assert not (X is None and base_model_logits is None)

        preds = [
            conformal_model.predict(X=X, base_model_logits=base_model_logits).astype('int')
            for conformal_model in self.conformal_models.values()
        ]

        return np.max(np.stack(preds, axis=0), axis=0)

    def _calibrate_icp_single_domain(self, preds: np.ndarray, y: np.ndarray, domain: Hashable) -> None:
        self.conformal_models[domain] = self.conformal_model_class(
            self.base_moedl, 
            num_classes=self.num_classes,
            pretraiend=self.pretrained,
            coverage=self.coverage,
            minimal_cnt_for_calibration=self.minimal_cnt_for_calibration,
            **self.conformal_model_kwargs
        )
        self.conformal_models[domain].calibrate_from_preds(preds, y)