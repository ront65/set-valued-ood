from evaluation.metrics.metric_abs import Metric
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pandas as pd
import seaborn as sns

class MinRecallAverage(Metric):

    name = "min_recall_average"


    def calc(model_preds : NDArray[Tuple[int, int]], y_true : NDArray[int], domains : NDArray[int]):
        # Define bin edges (from -4 to 4 with step size 0.5)
        bin_edges = np.arange(0, 1.025, 0.025)
        # Calculate bin midpoints
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        df = pd.DataFrame({
            'y' : y_true,
            'coverage' : model_preds[np.arange(len(model_preds)), y_true],
            'domain' : domains
        })
        per_domain_min_recall = (
            df
            .groupby(['domain', 'y'])
            ['coverage']
            .agg(['mean', 'size'])
            .loc[lambda df: df['size']>100]
            ['mean']
            .rename('coverage')
            .groupby('domain')
            .min()
        )
        return per_domain_min_recall.median()
    


    @classmethod
    def plot(cls, model_preds : NDArray[Tuple[int, int]], y_true : NDArray[int], domains : NDArray[int], ax, label):
        calc_res = cls.calc(model_preds, y_true, domains)
        calc_res.loc[0] = 0
        sns.histplot(data=calc_res.to_frame(),  y='coverage', label=label, ax=ax, cumulative=True, stat='probability',
                     element="step", fill=False, bins=100)
        ax.axhline(y=90/100, color='r', linestyle='--')