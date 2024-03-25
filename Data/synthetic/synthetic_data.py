import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import  Dataset
from typing import List, Tuple, Optional
from Data.utils import get_multi_domain_subsets_indices
import os 
from scipy.stats import special_ortho_group

SYNTHETIC_2_PARAMS = {
    'mean_of_label_0': np.array([0,0]),
    'label_1_mean_offset': np.array([0.3, 0.3]),
    'label_1_offset_direction': np.array([1,-1]),
    'min_offset_direction_change': -0.5,
    'max_offset_direction_change': 0.5,
    'features_std': [0.2, 0.2],
}

SYNTHETIC_10_PARAMS = {
    'mean_of_label_0': np.zeros(10),
    'label_1_mean_offset': 0.1*np.ones(10),
    'label_1_offset_direction': np.hstack([np.ones(5), -1*np.ones(5)]),
    'min_offset_direction_change': -0.5,
    'max_offset_direction_change': 0.5,
    'features_std': 0.2*np.ones(10),
}

SYNTHETIC_50_PARAMS = {
    'mean_of_label_0': np.zeros(50),
    'label_1_mean_offset': 0.05*np.ones(50),
    'label_1_offset_direction': np.hstack([np.ones(25), -1*np.ones(25)]),
    'min_offset_direction_change': -0.3,
    'max_offset_direction_change': 0.3,
    'features_std': 0.25*np.ones(50),
}


class SyntheticDataset(Dataset):

    def __init__(self, data: pd.DataFrame, features: List[str], y_col: str, domain_col:str, allow_log_data: bool) -> None:
        self._data = data
        self._features = features
        self._y_col = y_col
        self._domain_col = domain_col
        self.allow_log_data = allow_log_data

        self._X = torch.tensor(self._data[features].values, dtype=torch.float32)
        self._y = torch.tensor(self._data[y_col].values)
        self._domains = torch.tensor(self._data[domain_col].values)
    
    def __len__(self) -> None:
        return len(self._data)

    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx], self._domains[idx]
    
    @property
    def domains(self ) -> torch.Tensor:
        return self._domains

    @property
    def labels(self) -> torch.Tensor:
        return self._y

    def log_data(self, name: str, folder_path) -> Optional[str]:
        if self.allow_log_data:
            filename = f"{name}_data.npy"
            filepath = os.path.join(folder_path, filename)
            # np.save(filepath, self._data)
            self._data.to_csv(filepath)
            return filepath
        else:
            return None


def get_synthetic_train_test_subsets(
    data_params: dict, 
    allow_log_data: bool = False,
    num_train_domains: int = 20,
    num_ood_domains = None,
    train_domain_size: int = 1000, 
    test_domain_size: int = 1000,  
    CVC_split: bool = False,
    **kwargs,
):

    data_df, features, y_column, domains_column = _generate_data_of_multiple_domains_classification_chnage_on_line(
        size=10_000,
        num_of_domains=50,
        **data_params,
    )
    train_indices, in_domain_test_indices, ood_test_indices = get_multi_domain_subsets_indices(
        domains_array=data_df[domains_column].values,
        in_domain_domains=None,
        ood_domains=None,
        num_train_domains=num_train_domains,
        num_ood_domains=num_ood_domains,
        train_domain_size=train_domain_size, 
        test_domain_size=test_domain_size, 
        CVC_split=CVC_split,
        )

    print(f"Creating training data:")
    print(data_df.iloc[train_indices][domains_column].value_counts())
    train_subset = SyntheticDataset(data_df.iloc[train_indices], features, y_column, domains_column, allow_log_data)
    
    #--- Get in-domain test subset
    print(f"Creating in-domain-test data:")
    print(data_df.iloc[in_domain_test_indices][domains_column].value_counts())
    in_domain_test_subset = SyntheticDataset(data_df.iloc[in_domain_test_indices], features, y_column, domains_column, allow_log_data)
    
    #--- Get ood test subset
    if num_ood_domains is not None:
        print(f"Creating ood domains:")
        print(data_df.iloc[ood_test_indices][domains_column].value_counts())
        ood_test_subset = SyntheticDataset(data_df.iloc[ood_test_indices], features, y_column, domains_column, allow_log_data)
    else:
        ood_test_subset = None
    
    #### Return subsets
    print("returning datasets")
    return train_subset, in_domain_test_subset, ood_test_subset


def _generate_data_of_specific_domain_classification(
    size=10_000,
    domain_offset_of_label_1=[1],
    domain_mean_of_label_0=[0],
    features_std=[0.1],
    random_cov: bool = False,
):
    assert len(domain_mean_of_label_0) == len(domain_offset_of_label_1)
    assert len(domain_mean_of_label_0) == len(features_std)

    p = 0.5
    #Generate random bernouli variable with probability p
    y = np.random.binomial(n=1, p=p, size=int(size))

    if random_cov:
        D= np.diag(np.square(features_std))
        Q = special_ortho_group.rvs(len(features_std))
        cov = Q @ D @ Q.T
        features = (
            y.reshape(-1,1) * domain_offset_of_label_1 + domain_mean_of_label_0
            +
            np.random.multivariate_normal(
                mean = np.zeros(len(features_std)), 
                cov = cov, 
                size = int(size)
            )
        )
    else:
        features = [y * feature_offset + feature_mean_of_label_0 + np.random.normal(loc=0, scale=std, size=int(size))
            for feature_offset, feature_mean_of_label_0, std in zip(domain_offset_of_label_1, domain_mean_of_label_0, features_std)]

    return pd.concat(
        [
            pd.DataFrame({'y': y}),
            pd.DataFrame(features) if random_cov else pd.DataFrame(np.vstack(features)).T
        ],
        axis=1
    )

def _generate_data_of_multiple_domains_classification_chnage_on_line(
    size=10_000,
    num_of_domains=10,
    mean_of_label_0 = np.array([0,0]),
    label_1_mean_offset = np.array([0.25, 0.25]),
    label_1_offset_direction = np.array([1,-1]),
    min_offset_direction_change = -1,
    max_offset_direction_change = 1,
    features_std=[0.1, 0.1],
    random_cov: bool = False,
):
    """ Generate domains according to the following DGP:
        For each domain points of label 0 will be sampled around a common mean, and points of lavel 1 will
        be samples around a per-domain "offset". The different offsets will be sampled (one offset for each domain)
        uniformaly from a truncated line. Eventually, for each domaiin the points of label 1 will be sampled around
        a mean that equals (mean_of_label_0 + domain_label_1_offset), and for each domain domain_label_1_offset will be
        sampled from a truncated line. 
        The truncated line is defined by the input arguments as the closed interval:
        [label_1_mean_offset + label_1_offset_direction * min_offset_direction_change, label_1_mean_offset + label_1_offset_direction * max_offset_direction_change]
        Args:
            size: number of points to generate at each domain
            num_of_domains: Number of domains to generate
            mean_of_label_0: The mean for points of label 0. This is common for all domains.
            label_1_mean_offset: This is the center of the truncated line from which the per-domain offset for label 1
                                will be sampled.
            label_1_offset_direction: The direction of the truncated line from which the per-domain offset for label 1
                                will be sampled.
            min_offset_direction_change: This sets one end for the truncated line from which the per-domain offset for label 1
                                will be sampled. 
            max_offset_direction_change: This sets another end for the truncated line from which the per-domain offset for label 1
                                will be sampled. 
            features_std: The std for each feature. 
    """
    print("Generating random cov matrix for each domain")
    assert len(features_std) == len(label_1_mean_offset)
    assert len(mean_of_label_0) == len(features_std)

    offsets = [
        label_1_mean_offset + label_1_offset_direction * offset_direction_change
        for offset_direction_change in np.random.uniform(low=min_offset_direction_change, high=max_offset_direction_change, size=num_of_domains)
    ]

    data = pd.concat(
        [
            (
                _generate_data_of_specific_domain_classification(
                    size=size,
                    domain_offset_of_label_1=offset,
                    domain_mean_of_label_0=mean_of_label_0,
                    features_std=(
                        features_std + 
                        (
                            np.random.normal(loc=0, scale=0.05, size=len(features_std))
                            if random_cov else 0
                        )
                    ),
                    random_cov=random_cov,
                )
                .assign(
                    offset = lambda df: [offset]*df.shape[0],
                    mean_of_label_0 = lambda df: [mean_of_label_0]*df.shape[0],
                )
                .assign(domain=j)
            )
            for j, offset in enumerate(offsets)
        ],
        axis=0
    ).assign(**{f'feature_{i}_stds': feature_std for i, feature_std in enumerate(features_std)})

    return data, [i for i in range(len(features_std))], 'y', 'domain'
