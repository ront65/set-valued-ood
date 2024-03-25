from evaluation.metrics.metric_abs import Metric
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pandas as pd
import seaborn as sns

class MeanSize(Metric):

    name = "mean_size"

    @staticmethod
    def calc(model_preds : NDArray[Tuple[int, int]], y_true : NDArray[int], domains : NDArray[int]):
        df = (
                pd.DataFrame({
                'y' : y_true,
                'coverage' : model_preds[np.arange(len(model_preds)), y_true],
                'domain' : domains,
                'size': model_preds.sum(axis=1),
            })
            .assign(
                domain_label_count = lambda df: df.groupby(['domain', 'y'])['coverage'].transform('size')
            )
            .loc[lambda df: df['domain_label_count']>100]
        )
        per_domain_size = df.groupby('domain')['size'].mean()
        return pd.Series(per_domain_size.median(), name="mean_size")


    @classmethod
    def plot(cls, model_preds : NDArray[Tuple[int, int]], y_true : NDArray[int], domains : NDArray[int], ax, label):
        calc_res = cls.calc(model_preds, y_true, domains)
        data_to_plot = calc_res.to_frame().assign(__plot_label=label)
        sns.barplot(data = data_to_plot, x='__plot_label', y = 'mean_size', ax=ax)
