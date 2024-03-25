import numpy as np

from evaluation.metrics.metric_abs import Metric
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pandas as pd
from collections import defaultdict


CB_color_cycle = ['#8da0cb', '#fc8d62', '#b3b3b3', '#66c2a5', '#e78ac3', '#a65628' ]

class MinRecallVsSizeCross(Metric):

    name = "min_recall_vs_size_scatter"
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
        per_domain_size = df.groupby('domain')['size'].mean()
        combined_series = pd.Series({
            'median_size': per_domain_size.median(), 
            'median_min_recall': per_domain_min_recall.median(),
            'std_size': per_domain_size.std(),
            'std_min_recall': per_domain_min_recall.std(),
            'p10_size': np.percentile(per_domain_size, 25),
            'p10_min_recall': np.percentile(per_domain_min_recall, 25),
            'p90_size': np.percentile(per_domain_size, 75),
            'p90_min_recall': np.percentile(per_domain_min_recall, 75),
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
        var_type='std',
        coverage: float = 0.9,
    ):
        assert len(model_preds) == len(y_true) == len(domains)
        num_seeds = len(model_preds)
        calc_res = pd.concat([cls.calc(model_preds[i], y_true[i], domains[i]) for i in range(num_seeds)], axis=1).mean(axis=1)
        
        if var_type=='std':
            ax.errorbar(
                calc_res['median_size'], 
                calc_res['median_min_recall'], 
                xerr=calc_res['std_size'], 
                yerr=calc_res['std_min_recall'], 
                color = CB_color_cycle[cls.cnt[ax]],
                capsize=5, 
                capthick=2,
                label=f'Mean ± Std : {label}'
            )
        elif var_type=='percentiles':
            ax.plot([calc_res['p10_size'], calc_res['p90_size']], 
                    [calc_res['median_min_recall'], calc_res['median_min_recall']], 
                    color=CB_color_cycle[cls.cnt[ax]], 
                    linewidth=2,
                    marker='|'
            )
            ax.plot([calc_res['median_size'], calc_res['median_size']], 
                    [calc_res['p10_min_recall'], calc_res['p90_min_recall']], 
                    color=CB_color_cycle[cls.cnt[ax]], 
                    linewidth=2,
                    marker = '_'
            )

            # Plot the center point
            ax.plot(calc_res['median_size'], calc_res['median_min_recall'], 'o', color=CB_color_cycle[cls.cnt[ax]], markersize=8, label=label)
        
        ax.axhline(coverage, color='#e5c494', linestyle='dotted', alpha=1)
        cls.cnt[ax] += 1
