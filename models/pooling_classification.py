import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

class PoolingCdf__Classification:
    ACCURACY = 'acc'
    PER_Y_RECALL = 'per_y_recall'

    def __init__(
            self, 
            base_moedl, 
            num_classes: int, 
            pretrained: bool = False, 
            coverage: float = 0.9, 
            coverage_type: str = 'per_y_recall', 
            minimal_cnt_for_calibration: int = 1,
        ):
        self.interval_size = None
        self.base_moedl = base_moedl
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.coverage = coverage
        assert coverage_type in [self.ACCURACY, self.PER_Y_RECALL]
        self.coverage_type = coverage_type
        self.minimal_cnt_for_calibration = minimal_cnt_for_calibration

    def fit(self, X: ArrayLike, y: ArrayLike, domains: ArrayLike):
        if not self.pretrained:
            self.base_moedl.fit(X, y)

    def calibrate(self, X: ArrayLike, y: np.ndarray, domains: np.ndarray) -> None:
        assert y.ndim == domains.ndim == 1
        assert X.shape[0] == y.shape[0] == domains.shape[0]

        preds = self.base_moedl.predict_proba(X) # Preds should be np.array
        self.calibrate_from_preds(preds, y, domains)

    def calibrate_from_preds(self, preds: np.ndarray, y: np.ndarray, domains: np.ndarray) -> None:
        assert preds.ndim == 2 and y.ndim == 1 and domains.ndim == 1
        assert preds.shape[0] == y.shape[0] == domains.shape[0]
        assert preds.shape[1] == self.num_classes

        conformity_score__cal = preds[np.arange(len(y)), y]
        if self.coverage_type == self.ACCURACY:
            self.threshold = self._get_threshold(conformity_score__cal, domains)
            assert isinstance(self.threshold, float)
        elif self.coverage_type == self.PER_Y_RECALL:
            self.threshold = np.zeros(self.num_classes)
            for clss in np.unique(y):
                self.threshold[clss] = self._get_threshold(conformity_score__cal[y == clss], domains[y == clss])
                
        return self

    def predict(self, X: ArrayLike, base_model_logits: Optional[np.ndarray] = None) -> np.ndarray:
        assert X is None or base_model_logits is None
        assert not (X is None and base_model_logits is None)

        probs = self.base_moedl.predict_proba(X) if base_model_logits is None else base_model_logits
        return probs >= self.threshold


    def _compute_residuals_cdf(self, data):
        """Compute the CDF function for a given list of numbers."""
        sorted_data = np.sort(data)
        n = len(data)

        def cdf(x):
            return np.searchsorted(sorted_data, x, side='right') / n

        return cdf


    def _get_threshold(self, residuals: pd.Series, domains: pd.Series):
        """Find the smallest t such that the average CDF(t) over all lists is larger than threshold."""
        # Find unique domains
        unique_domains = np.unique(domains)

        ## We want the top % of residuals, which is equal to bottom % of -1*residuals
        minus_residuals = -1 *  residuals
        # Compute CDF functions for all unique domains
        cdfs = [self._compute_residuals_cdf(minus_residuals[domains == domain]) for domain in unique_domains if sum(domains == domain) >= self.minimal_cnt_for_calibration]

        # Find unique values to check as potential t values
        unique_values = np.unique(minus_residuals)

        # Iterate over all unique values to find the smallest t
        for t in unique_values:
            avg_cdf_value = np.mean([cdf(t) for cdf in cdfs])
            if avg_cdf_value > self.coverage:
                ## t is a value in minus_residuals that gives bottom %. We multiply by -1 to get the value
                ## in residuals that gives the top %.
                return -1* t
        return None  # In case no such t is found
    
