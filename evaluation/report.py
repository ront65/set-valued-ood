from models.robust_conformal import RobustConformal
from models.pooling_classification import PoolingCdf__Classification

import numpy as np
import pandas as pd
from typing import List
import wandb
import shutil
import matplotlib.pyplot as plt
from evaluation.metrics import (
    MeanSize,
    MinRecallAverage,
    MinRecallVsSizeCross,
    DomainCorrectnessBarPlot,
    MinRecallBoxPlot,
)
from collections import defaultdict
from experiments.utils import DataDumpTypes

DOWANLOAD_PATH='/vol/scratch/rontsibulsky/set-valued-domain-generalization/artifacts'

class REPORTER:
    def __init__(self, project: str, runs:  dict[str, list[str]], data_sets:List[str], epoch: int, num_classes: int, coverage: float = 0.9, add_conformals: bool = True):
        self.num_seeds = len(runs['standard'])
        for runs_list in runs.values():
            assert len(runs_list) == self.num_seeds
        self.project = project
        self.epoch = epoch
        self.num_classes = num_classes
        self.coverage = coverage
        self._get_data(runs, data_sets)
        if add_conformals:
            self._add_robust_conformal_model_to_data()
            self._add_pooling_cdfs_TrainC_to_data()
            self._add_pooling_cdfs_CVC_to_data()
        

    
    def _get_data(self, runs: dict[str, list[str]], data_sets:List[str]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        data = {
            model:
            {
                i : {
                    data_set: self._get_run_data_set_data(run=run_id, data_set=data_set)
                    for data_set in data_sets
                }
                for i, run_id in enumerate(runs_list)
            }
            for model, runs_list in runs.items()
        }
        self.data = data

    def _add_robust_conformal_model_to_data(self) -> None:
        conformals = [
            RobustConformal(
                base_moedl = None,
                num_classes = self.num_classes,
                conformal_model_kwargs = dict(),
                pretrained = True, 
                coverage=self.coverage,
                minimal_cnt_for_calibration=100,
            )
            for i in range(self.num_seeds)
        ]
        for i in range(self.num_seeds):
            conformals[i].calibrate_from_preds(
                preds=self.data['standard'][i]['train'][DataDumpTypes.LOGIT_PRED], 
                y=self.data['standard'][i]['train'][DataDumpTypes.LABEL], 
                domains=self.data['standard'][i]['train'][DataDumpTypes.DOMAIN]
            ) 
        self.data['conformal'] = {
            i : {
                data_set: {
                    DataDumpTypes.CLASS_PRED: conformals[i].predict(
                        X = None, 
                        base_model_logits = self.data['standard'][i][data_set][DataDumpTypes.LOGIT_PRED]
                    ),
                    DataDumpTypes.LOGIT_PRED: None,
                    DataDumpTypes.LABEL: self.data['standard'][i][data_set][DataDumpTypes.LABEL],
                    DataDumpTypes.DOMAIN: self.data['standard'][i][data_set][DataDumpTypes.DOMAIN],
                }
                for data_set in self.data['standard'][i].keys()
            }
            for i in range(self.num_seeds)
        }

    def _add_pooling_cdfs_TrainC_to_data(self) -> None:
        pool_cdfs = [
            PoolingCdf__Classification(
                base_moedl = None,
                num_classes = self.num_classes,
                pretrained = True, 
                coverage=self.coverage,
                coverage_type='per_y_recall', 
                minimal_cnt_for_calibration=100,
            )
            for i in range(self.num_seeds)
        ]
        for i in range(self.num_seeds):
            pool_cdfs[i].calibrate_from_preds(
                preds=self.data['standard'][i]['train'][DataDumpTypes.LOGIT_PRED], 
                y=self.data['standard'][i]['train'][DataDumpTypes.LABEL], 
                domains=self.data['standard'][i]['train'][DataDumpTypes.DOMAIN]
            ) 
        self.data['poolingCDF_TrainC'] = {
            i:{
                data_set: {
                    DataDumpTypes.CLASS_PRED: pool_cdfs[i].predict(
                        X = None, 
                        base_model_logits = self.data['standard'][i][data_set][DataDumpTypes.LOGIT_PRED]
                    ),
                    DataDumpTypes.LOGIT_PRED: None,
                    DataDumpTypes.LABEL: self.data['standard'][i][data_set][DataDumpTypes.LABEL],
                    DataDumpTypes.DOMAIN: self.data['standard'][i][data_set][DataDumpTypes.DOMAIN],
                }
                for data_set in self.data['standard'][i].keys()
            }
            for i in range(self.num_seeds)
        }

    def _add_pooling_cdfs_CVC_to_data(self) -> None:
        pool_cdfs = [
            PoolingCdf__Classification(
                base_moedl = None,
                num_classes = self.num_classes,
                pretrained = True, 
                coverage=self.coverage,
                coverage_type='per_y_recall', 
                minimal_cnt_for_calibration=100,
            )
            for i in range(self.num_seeds)
        ]
        for i in range(self.num_seeds):
            standard_run_train_domain = np.unique(self.data['standard'][i]['train'][DataDumpTypes.DOMAIN])
            CVC_run_train_domains = np.unique(self.data['poolCDF_CVC'][i]['train'][DataDumpTypes.DOMAIN])
            CVC_run_ood_domains = np.unique(self.data['poolCDF_CVC'][i]['ood-test'][DataDumpTypes.DOMAIN])
            calibration_domains = np.array(list(set(standard_run_train_domain) - set(CVC_run_train_domains)))

            assert set(CVC_run_train_domains) < set(standard_run_train_domain)
            assert set(calibration_domains) < set(CVC_run_ood_domains)
            calibration_mask = np.isin(
                self.data['poolCDF_CVC'][i]['ood-test'][DataDumpTypes.DOMAIN],
                calibration_domains
            )

            pool_cdfs[i].calibrate_from_preds(
                preds=self.data['poolCDF_CVC'][i]['ood-test'][DataDumpTypes.LOGIT_PRED][calibration_mask], 
                y=self.data['poolCDF_CVC'][i]['ood-test'][DataDumpTypes.LABEL][calibration_mask], 
                domains=self.data['poolCDF_CVC'][i]['ood-test'][DataDumpTypes.DOMAIN][calibration_mask]
            ) 
        self.data['poolingCDF_CVC'] = {
            i:{
                data_set: {
                    DataDumpTypes.CLASS_PRED: pool_cdfs[i].predict(
                        X = None, 
                        base_model_logits = self.data['standard'][i][data_set][DataDumpTypes.LOGIT_PRED]
                    ),
                    DataDumpTypes.LOGIT_PRED: None,
                    DataDumpTypes.LABEL: self.data['standard'][i][data_set][DataDumpTypes.LABEL],
                    DataDumpTypes.DOMAIN: self.data['standard'][i][data_set][DataDumpTypes.DOMAIN],
                }
                for data_set in self.data['standard'][i].keys()
            }
            for i in range(self.num_seeds)
        }


    def plot_recall_vs_size_cross(self, models: dict[str,str], data_sets: List[str], var_type='percentile'):
        fig, axes = plt.subplots(1,len(data_sets), figsize=(5*len(data_sets),5), sharey=True, sharex=True)

        dats_sets_axes_map = {
            data_set:i
            for i, data_set in enumerate(data_sets)
        }
        for data_set in data_sets:
            ax = (
                axes[dats_sets_axes_map[data_set]]
                if len(data_sets) > 1 
                else axes
            )
            for model in models.keys():
                # print(data_set, run)
                MinRecallVsSizeCross.plot(
                    model_preds=[self.data[model][i][data_set][DataDumpTypes.CLASS_PRED] for i in range(self.num_seeds)], 
                    y_true=[self.data[model][i][data_set][DataDumpTypes.LABEL] for i in range(self.num_seeds)], 
                    domains=[self.data[model][i][data_set][DataDumpTypes.DOMAIN] for i in range(self.num_seeds)],    
                    ax=ax, 
                    label=models[model],
                    var_type=var_type,
                    coverage=self.coverage,
                )
            ax.set_title(data_set)
            ax.set_xlabel('Size')
            ax.set_ylabel('Min Recall')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        print(xlim, ylim)
        ax.plot([0, self.num_classes], [0, 1], linestyle='--', color='#a6d854')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()
        plt.show()


    def plot_recall_boxplot(self, models: dict[str,str], data_sets: List[str],):
        fig, axes = plt.subplots(1,len(data_sets), figsize=(5*len(data_sets),5), sharey=True, sharex=True)

        dats_sets_axes_map = {
            data_set:i
            for i, data_set in enumerate(data_sets)
        }
        for data_set in data_sets:
            ax = (
                axes[dats_sets_axes_map[data_set]]
                if len(data_sets) > 1 
                else axes
            )
            labels = []
            positions = []
            for i,model in enumerate(models.keys()):
                MinRecallBoxPlot.plot(
                    model_preds=[self.data[model][i][data_set][DataDumpTypes.CLASS_PRED] for i in range(self.num_seeds)], 
                    y_true=[self.data[model][i][data_set][DataDumpTypes.LABEL] for i in range(self.num_seeds)], 
                    domains=[self.data[model][i][data_set][DataDumpTypes.DOMAIN] for i in range(self.num_seeds)],    
                    ax=ax, 
                    label=models[model],
                    coverage=self.coverage,
                )
                labels.append(models[model])
                positions.append(i+1)
            ax.set_title(data_set)
            ax.set_xlabel('Min Recall')
            ax.set_xticks(positions)  # Match the positions used in the `bxp` call
            ax.set_xticklabels(labels)  # Set the corresponding labels

            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
    

    def plot_correctness_barplot(self, models: dict[str,str], data_sets: List[str]):
        fig, axes = plt.subplots(1,len(data_sets), figsize=(5*len(data_sets),5), sharey=True, sharex=True)

        dats_sets_axes_map = {
            data_set:i
            for i, data_set in enumerate(data_sets)
        }
        for data_set in data_sets:
            ax = (
                axes[dats_sets_axes_map[data_set]]
                if len(data_sets) > 1 
                else axes
            )
            for model in models.keys():
                DomainCorrectnessBarPlot.plot(
                    model_preds=[self.data[model][i][data_set][DataDumpTypes.CLASS_PRED] for i in range(self.num_seeds)], 
                    y_true=[self.data[model][i][data_set][DataDumpTypes.LABEL] for i in range(self.num_seeds)], 
                    domains=[self.data[model][i][data_set][DataDumpTypes.DOMAIN] for i in range(self.num_seeds)],    
                    ax=ax, 
                    label=models[model],
                    coverage=self.coverage
                )
            ax.set_title(data_set)
            ax.set_xlabel('Recall > 90% PCTG')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()


    def plot_table(self, models: dict[str,str], data_sets: List[str]) -> pd.DataFrame:
    
        d = defaultdict(lambda: defaultdict(list))
        for data_set in data_sets:
            for model in models.keys():
                per_seed_sizes = [
                    MeanSize.calc(
                        model_preds=self.data[model][i][data_set][DataDumpTypes.CLASS_PRED], 
                        y_true=self.data[model][i][data_set][DataDumpTypes.LABEL], 
                        domains=self.data[model][i][data_set][DataDumpTypes.DOMAIN], 
                    )
                    for i in range(self.num_seeds)
                ]
                mean_size = np.mean(per_seed_sizes).round(3)
                std_size = np.std(per_seed_sizes).round(3)
                d[(data_set, 'Median size')][models[model]] = (mean_size, std_size)


        for data_set in data_sets:
            for model in models.keys():
                per_seed_correctness = [
                    DomainCorrectnessBarPlot.calc(
                        model_preds=self.data[model][i][data_set][DataDumpTypes.CLASS_PRED], 
                        y_true=self.data[model][i][data_set][DataDumpTypes.LABEL], 
                        domains=self.data[model][i][data_set][DataDumpTypes.DOMAIN],    
                        coverage = self.coverage
                    )
                    for i in range(self.num_seeds)
                ]
                mean_correctness = np.mean(per_seed_correctness).round(3)
                std_correctness = np.std(per_seed_correctness).round(3)
                d[(data_set, 'correctness')][models[model]] = (mean_correctness, std_correctness)


        for data_set in data_sets:
            for model in models.keys():
                per_seed_recall = [
                    MinRecallAverage.calc(
                        model_preds=self.data[model][i][data_set][DataDumpTypes.CLASS_PRED], 
                        y_true=self.data[model][i][data_set][DataDumpTypes.LABEL], 
                        domains=self.data[model][i][data_set][DataDumpTypes.DOMAIN],    
                    )
                    for i in range(self.num_seeds)
                ]
                mean_recall = np.mean(per_seed_recall).round(3)
                std_recall = np.std(per_seed_recall).round(3)
                d[(data_set, 'average min_recall')][models[model]] = (mean_recall, std_recall)

        return pd.DataFrame(d)[['train', 'in-domain-test', 'ood-test']]


    def _get_run_data_set_data(self, run, data_set):
        try:
            api = wandb.Api()
            run = api.run(f"{self.project}/{run}")
            download_path = DOWANLOAD_PATH
            for artifact in run.logged_artifacts():
                if f"preds_{data_set}" in artifact.name:
                    artifact = run.use_artifact(artifact)
                    artifact_dir = artifact.download(download_path)

            def load_data(data_type):
                return np.load(f"{artifact_dir}/{data_set}_{data_type.value}_epoch{self.epoch}__{run.id}.npy")

            res = {
                DataDumpTypes.CLASS_PRED: load_data(DataDumpTypes.CLASS_PRED),
                DataDumpTypes.LOGIT_PRED: load_data(DataDumpTypes.LOGIT_PRED),
                DataDumpTypes.LABEL: load_data(DataDumpTypes.LABEL),
                DataDumpTypes.DOMAIN: load_data(DataDumpTypes.DOMAIN),
            }
        finally:
            shutil.rmtree(download_path)

        return res
    