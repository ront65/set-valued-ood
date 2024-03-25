import random
import torch
from typing import List
from torch.utils.data import Sampler
import numpy as np
from enum import Enum

DUMP_FOLDER = '/vol/scratch/rontsibulsky/set-valued-domain-generalization/camelyon/runs_dump'

class DataDumpTypes(Enum):
    CLASS_PRED = 'class_pred'
    LOGIT_PRED = 'logit_pred'
    LABEL = 'label'
    DOMAIN = 'domain'
    CORRECT = 'correct'

class CustomBatchSampler(Sampler):

    def __init__(self, domains: torch.Tensor, mini_batch_size: int, domains_per_batch: int = 4):
        self.domains = domains
        self.mini_batch_size = mini_batch_size
        self.domains_per_batch = domains_per_batch

        #Create domain to indices mapping
        self.unique_domains = np.unique(domains)
        self.domain_to_indices = {
            domain: np.where(domains == domain)[0].tolist()
            for domain in self.unique_domains
        }
        self.batches = self._create_batches()


    def _create_minibatches(self, domain_indices: List[int]) -> List[List[int]]:
        indices = domain_indices.copy()
        np.random.shuffle(indices)

        num_complete_batches = len(indices) // self.mini_batch_size
        minibatches = [
            indices[i:i + self.mini_batch_size]
            for i in range(0, num_complete_batches*self.mini_batch_size, self.mini_batch_size)
        ]
        return minibatches


    def _create_batches(self,) -> List[torch.Tensor]:
        all_minibatches = []

        # Create batches for each label
        for domain in self.unique_domains:
            domain_indices = self.domain_to_indices[domain]
            domain_minibatches = self._create_minibatches(domain_indices)
            all_minibatches.extend(domain_minibatches)
        
        # Shuffle the indices
        random.shuffle(all_minibatches)

        batches = []
        # Split indices into batches of specified size
        for i in range(0, len(all_minibatches), self.domains_per_batch):
            if i + self.domains_per_batch <= len(all_minibatches):
                batch = []
                for minibatch in all_minibatches[i : i + self.domains_per_batch]:
                    batch.extend(minibatch)
                batches.append(batch)
        
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


