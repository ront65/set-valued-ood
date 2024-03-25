import numpy as np
import pandas as pd
import torch
from wilds.datasets.wilds_dataset import WILDSSubset, WILDSDataset
from torch.utils.data import  Dataset
from torchvision import models
from typing import Tuple, List, Optional
import os

from Data.utils import get_multi_domain_subsets_indices


class WildsDatasetWrapper(Dataset):

    def __init__(self, wilds_dataset):
        self.wilds_dataset = wilds_dataset
        self.len = len(wilds_dataset)
    
    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.wilds_dataset[idx]

    @property
    def labels(self) -> torch.Tensor:
        return self.wilds_dataset.y_array
    

    @property
    def domains(self) -> torch.Tensor:
        return self.wilds_dataset.metadata_array

    @property
    def indices(self):
        try:
            return self.wilds_dataset.indices
        except:
            "No property indices in dataset"

    def log_data(self, name: str, folder_path) -> str:
        filename = f"{name}_indices.npy"
        filepath = os.path.join(folder_path, filename)
        np.save(filepath, self.indicies)
        return filepath

    
class PreloadWildsDataset(Dataset):
    def __init__(self, wilds_dataset, transform = None):
        self.wilds_dataset = wilds_dataset
        self.len = len(wilds_dataset)

        all_items = [(image, target, domain) for (image, target, domain) in wilds_dataset]
        self._images, self._targets, self._domains = tuple(
            map(
                torch.stack, 
                zip(*all_items)
            )
        )
        
        assert isinstance(self._targets, torch.Tensor) and self._targets.dim() == 1 and self._targets.shape[0] == self.len
        assert isinstance(self._domains, torch.Tensor) and self._domains.dim() == 1 and self._domains.shape[0] == self.len
        assert isinstance(self._images, torch.Tensor) and self._images.shape[0] == self.len
        self.transform = transform

    def __len__(self) -> int:
        return self.len
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.transform is None:
            return self._images[idx], self._targets[idx], self._domains[idx]
        else:
            return self.transform(self._images[idx]), self._targets[idx], self._domains[idx]

    @property
    def labels(self) -> torch.Tensor:
        return self._targets
    

    @property
    def domains(self) -> torch.Tensor:
        return self._domains


    @property
    def indices(self):
        try:
            return self.wilds_dataset.indices
        except:
            "No property indices in dataset"
    

    def log_data(self, name: str, folder_path) -> str:
        filename = f"{name}_indices.npy"
        filepath = os.path.join(folder_path, filename)
        np.save(filepath, self.indices)
        return filepath


def get_wilds_train_test_subsets(
    main_data_set: WILDSDataset,
    in_domain_domains: Optional[List[str]], 
    ood_domains = None, 
    num_train_domains = None,
    num_ood_domains = None,
    train_domain_size: int = 1000, 
    test_domain_size: int = 1000, 
    use_preloaded_dataset = True,
    transform = models.ResNet18_Weights.DEFAULT.transforms(),
    CVC_split: bool = False,
    **kwargs
) -> Tuple[WILDSDataset, WILDSDataset, Optional[WILDSDataset]]:
    
    domains_array = main_data_set.metadata_array.numpy()

    train_indices, in_domain_test_indices, ood_test_indices = get_multi_domain_subsets_indices(
        domains_array=domains_array,
        in_domain_domains=in_domain_domains, 
        ood_domains=ood_domains, 
        num_train_domains=num_train_domains,
        num_ood_domains=num_ood_domains,
        train_domain_size=train_domain_size, 
        test_domain_size=test_domain_size, 
        CVC_split = CVC_split,
    )
    #### Get all subsets    
    
    #--- Get train subset
    print(f"Creating training data:")
    print(pd.Series(domains_array[train_indices]).value_counts())
    train_subset = get_subset(main_dataset=main_data_set, indices=train_indices, transform=transform, use_preloaded_dataset=use_preloaded_dataset)
    
    #--- Get in-domain test subset
    print(f"Creating in-domain-test data:")
    assert len(in_domain_test_indices) > 0, "In domain test must bot be empty!"
    print(pd.Series(domains_array[in_domain_test_indices]).value_counts())
    in_domain_test_subset = get_subset(main_dataset=main_data_set, indices=in_domain_test_indices, transform=transform, use_preloaded_dataset=use_preloaded_dataset)
    
    #--- Get ood test subset
    if ood_test_indices is not None:
        print(f"Creating ood domains:")
        print(pd.Series(domains_array[ood_test_indices]).value_counts())
        ood_test_subset = get_subset(main_dataset=main_data_set, indices=ood_test_indices, transform=transform, use_preloaded_dataset=use_preloaded_dataset)
    else:
        ood_test_subset = None
    
    #### Return subsets
    print("returning datasets")
    return train_subset, in_domain_test_subset, ood_test_subset




def get_subset(main_dataset: WILDSDataset, indices: np.ndarray, transform, use_preloaded_dataset: bool) -> Dataset:
    res = WILDSSubset(
        main_dataset,
        indices=indices,
        transform = None if use_preloaded_dataset else transform
    )

    if use_preloaded_dataset:
        res = PreloadWildsDataset(res, transform = transform)
    else:
        res = WildsDatasetWrapper(res)

    return res
