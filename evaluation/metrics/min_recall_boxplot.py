import numpy as np

from evaluation.metrics.metric_abs import Metric
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pandas as pd
from collections import defaultdict


CB_color_cycle = ['#8da0cb', '#fc8d62', '#b3b3b3', '#66c2a5', '#e78ac3', ]

class MinRecallBoxPlot(Metric):

    name = "min_recall_boxplot"
    cnt = defaultdict(lambda: 0)
    
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
        per_domain_min_recall = df.groupby(['domain', 'y'])['coverage'].mean().groupby('domain').min()

        q1 = np.percentile(per_domain_min_recall, 25)
        q3 = np.percentile(per_domain_min_recall, 75)
        median = np.median(per_domain_min_recall)
        iqr = q3 - q1
        whisker_low = q1 - 1.5 * iqr
        whisker_high = q3 + 1.5 * iqr

        combined_series = pd.Series({
            'median_min_recall': median,
            'q1_min_recall': q1,
            'q3_min_recall': q3,
            'whisker_low_min_recall': whisker_low,
            'whisker_high_min_recall': whisker_high,

        })
        return combined_series


    @classmethod
    def plot(
        cls, 
        model_preds : list[NDArray[Tuple[int, int]]], 
        y_true : list[NDArray[int]],
        domains : list[NDArray[int]], 
        ax, 
        label, 
        coverage: float = 0.9,
    ):
        assert len(model_preds) == len(y_true) == len(domains)
        num_seeds = len(model_preds)
        calc_res = pd.concat([cls.calc(model_preds[i], y_true[i], domains[i]) for i in range(num_seeds)], axis=1).mean(axis=1)        

        ax.bxp(
            [{
                'med': calc_res['median_min_recall'],
                'q1': calc_res['q1_min_recall'],
                'q3': calc_res['q3_min_recall'],
                'whislo': calc_res['whisker_low_min_recall'],
                'whishi': calc_res['whisker_high_min_recall'],
            }], 
            showfliers=False,
            positions = [cls.cnt[ax] + 1]
        )

        
        ax.axhline(coverage, color='#e5c494', linestyle='dotted', alpha=1)
        cls.cnt[ax] += 1
