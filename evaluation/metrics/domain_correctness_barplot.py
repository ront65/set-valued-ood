import numpy as np
import matplotlib.pyplot as plt

from evaluation.metrics.metric_abs import Metric
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pandas as pd
from collections import defaultdict


CB_color_cycle = ['#8da0cb', '#fc8d62', '#b3b3b3', '#66c2a5', '#e78ac3', '#a65628' ]

class DomainCorrectnessBarPlot(Metric):

    name = "domain_correctness_barplot"
    cnt = defaultdict(lambda: 0)
    
    def calc(model_preds : NDArray[Tuple[int, int]], y_true : NDArray[int], domains : NDArray[int], coverage: float):
        df = (
                pd.DataFrame({
                'y' : y_true,
                'coverage' : model_preds[np.arange(len(model_preds)), y_true],
                'domain' : domains,
            })
            .assign(
                domain_label_count = lambda df: df.groupby(['domain', 'y'])['coverage'].transform('size')
            )
            .loc[lambda df: df['domain_label_count']>100]
        )
        per_domain_min_recall = df.groupby(['domain', 'y'])['coverage'].mean().groupby('domain').min()

        res = pd.Series({
            'correctness_percentage': (per_domain_min_recall >= coverage).mean(),
        })

        return res


    @classmethod
    def plot(
        cls, 
        model_preds : list[NDArray[Tuple[int, int]]], 
        y_true : list[NDArray[int]],
        domains : list[NDArray[int]], 
        ax, 
        label, 
        var_type='std',
        coverage: float = 0.9,
    ):
        assert len(model_preds) == len(y_true) == len(domains)
        num_seeds = len(model_preds)
        calc_res = pd.concat([cls.calc(model_preds[i], y_true[i], domains[i], coverage=coverage) for i in range(num_seeds)], axis=1).mean(axis=1)
        
        ax.bar(x = cls.cnt[ax] + 0.5, height = calc_res['correctness_percentage'].mean(), label = label, color=CB_color_cycle[cls.cnt[ax]])
        
        cls.cnt[ax] += 1
