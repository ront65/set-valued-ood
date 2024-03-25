from sklearn.model_selection import train_test_split
import numpy as np
from typing import Protocol, Optional, List, Tuple
import torch

class MultiDomainsDataset(Protocol):

    @property
    def domains(self) -> torch.Tensor: ...

    @property
    def labels(self)  -> torch.Tensor:...


def get_multi_domain_subsets_indices(
    domains_array: np.ndarray,
    in_domain_domains: Optional[List[str]], 
    ood_domains = None, 
    num_train_domains = None,
    num_ood_domains = None,
    train_domain_size: int = 1000, 
    test_domain_size: int = 1000, 
    CVC_split: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #### Assure inputs are valid
    assert in_domain_domains or num_train_domains, "Need to specify either in-domain domains, or a number of domains to draw for in-domain"
    assert not (in_domain_domains and num_train_domains), "Cannot specify both in-domain domains, and a number of domains to draw for in-domain"
    
    if not in_domain_domains:
        assert not ood_domains, "cannot specify ood domains without specifying in-domain domains"

    if not num_train_domains:
        assert not num_ood_domains, "cannot specify num ood domains without specifying num in-domain domains"

    #### Get in-domain and ood domains
    if not in_domain_domains:
        in_domain_domains, ood_domains = _get_in_domain_and_ood_domains(
            domains_array=domains_array,
            num_train_domains=num_train_domains,
            num_ood_domains=num_ood_domains
        )
    
    #### Get indices for all subsets
    #--- Get train and in-domain test indices
    train_indices, in_domain_test_indices = _get_in_domain_indices(
        domains_array=domains_array,
        in_domain_domains=in_domain_domains,
        train_domain_size=train_domain_size,
        test_domain_size=test_domain_size,
    )


    #--- Get ood test indices
    ood_test_indices = None
    if ood_domains is not None:
        ood_test_indices = _get_ood_indices(
            domains_array=domains_array,
            ood_domains=ood_domains,
            test_domain_size=test_domain_size,
        )

    ## If using CVC split, move half of the train domains  to ood.
    if CVC_split:
        domains_to_remove_from_train = np.random.choice(
            in_domain_domains, 
            size = len(in_domain_domains)//2, 
            replace=False
        )
        indices_to_remove_from_train_mask = np.isin(domains_array[train_indices], domains_to_remove_from_train)
        indices_to_remove_from_train = train_indices[indices_to_remove_from_train_mask]

        train_indices = train_indices[~indices_to_remove_from_train_mask]

        ood_test_indices = np.array([]) if ood_test_indices is None else ood_test_indices
        ood_test_indices = np.concatenate([np.array(ood_test_indices), indices_to_remove_from_train])

    return train_indices, in_domain_test_indices, ood_test_indices


def _get_in_domain_and_ood_domains(
    domains_array: np.ndarray, 
    num_train_domains: int, 
    num_ood_domains: Optional[int]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    # all_patients = dataset._metadata_df['patient'].unique()
    unique_domains = np.unique(domains_array)
    in_domain_domains, ood_domains_temp = train_test_split(
        unique_domains, 
        train_size=num_train_domains, 
        test_size=num_ood_domains, 
        random_state=np.random.get_state()[1][0]
    )

    ood_domains = ood_domains_temp if num_ood_domains else None
    return in_domain_domains, ood_domains



def _get_in_domain_indices(
    domains_array: np.ndarray, 
    in_domain_domains: np.ndarray, 
    train_domain_size: int, 
    test_domain_size: int,
) -> Tuple[np.ndarray, np.ndarray]:

    train_indices = []
    in_domain_test_indices = []
    for domain in in_domain_domains:
        domain_indices = np.where(domains_array == domain)[0]
        train_size=min(train_domain_size, len(domain_indices))
        test_size=min(test_domain_size, len(domain_indices) - train_size)
        if test_size == 0:
            domain_train_indices = np.array(domain_indices)
            domain_test_indices = np.array([])
        else:
            domain_train_indices, domain_test_indices = train_test_split(
                domain_indices, 
                train_size=train_size,
                test_size=test_size, 
                random_state=np.random.get_state()[1][0]
            )
        train_indices.extend(domain_train_indices)
        in_domain_test_indices.extend(domain_test_indices)
    
    return np.array(train_indices), np.array(in_domain_test_indices)



def _get_ood_indices(
    domains_array: np.ndarray, 
    ood_domains: np.ndarray, 
    test_domain_size: int,
) -> np.ndarray:

    ood_test_indices = []
    for domain in ood_domains:
        domain_indeces = np.where(domains_array == domain)[0]
        size=min(test_domain_size, len(domain_indeces))

        domain_test_indeces = np.random.choice(domain_indeces, size=size, replace=False)
        ood_test_indices.extend(domain_test_indeces)
        
    return np.array(ood_test_indices)