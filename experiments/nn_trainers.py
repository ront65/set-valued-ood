import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Data.utils import MultiDomainsDataset
import random

import time
import wandb
from experiments.utils import CustomBatchSampler
from typing import Tuple, List, Union, Dict
import os
from experiments.utils import CustomBatchSampler, DUMP_FOLDER, DataDumpTypes
from typing import Optional

class BasicNNTrainer:

    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        train_dataset: MultiDomainsDataset, 
        in_domain_test_dataset: MultiDomainsDataset, 
        ood_test_dataset: MultiDomainsDataset, 
        batch_size: int = 64,
        num_epochs: int = 10, 
        use_wandb: bool = False,
        log_every: int = 10,
    ):  
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_dataset = train_dataset
        self.in_domain_test_dataset = in_domain_test_dataset
        self.ood_test_dataset = ood_test_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_wandb = use_wandb
        self.log_every = log_every

        self.model.to(self.device)
        self._create_dataloaders(batch_size)

        self.classes: Dict[Union(float, int), int] = None
        self.domains: Dict[Union(float, int), int] = None
        self._index_classes_and_domains(train_dataset)
        assert self.classes_indeces_match_classes_values


    def train_model(self,) -> None:
        if self.use_wandb:
            wandb.watch(self.model, self.criterion, log="all", log_freq=10)

        for epoch in range(self.num_epochs):
            self._run_epoch(epoch)
            self._post_epoch(
                sets_to_eval = ['train', 'in-domain-test', 'ood-test'],
                epoch = epoch,
                last_epoch = (epoch == self.num_epochs - 1)
            )
        
        print('Finished Training')
    
    
    def save_model(self, folder_path: str, filename: str = 'model_state_dict.pth'):
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Construct the full path to save the model
        filepath = os.path.join(folder_path, filename)

        # Save the model's state dictionary
        torch.save(self.model.state_dict(), filepath)

        print(f'Model saved to {filepath}')

        if self.use_wandb:
            # Create an artifact
            model_artifact = wandb.Artifact('model-weights', type='model')

            # Add the weights file to the artifact
            model_artifact.add_file(filepath)

            # Log the artifact to WandB
            wandb.log_artifact(model_artifact)

    
    def _create_dataloaders(self, batch_size: int) -> None:
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.in_domain_test_dataloader = DataLoader(self.in_domain_test_dataset, batch_size=batch_size, shuffle=False)
        self.ood_test_dataloader = DataLoader(self.ood_test_dataset, batch_size=batch_size, shuffle=False)
    

    def _index_classes_and_domains(self, train_dataset: MultiDomainsDataset) -> None:
        self.classes = {clss: i for i, clss in enumerate(np.unique(train_dataset.labels.numpy()))}
        self.classes_indeces_match_classes_values = all([(cls_idx == cls_val) for  cls_val, cls_idx in self.classes.items()])
        
        self.domains = {domain: i for i, domain in enumerate(np.unique(train_dataset.domains.numpy()))}

        self.num_classes = len(self.classes.keys())
        self.num_domains = len(self.domains.keys()) 


    def _run_epoch(self, epoch: int) -> None:
        self.model.train()
        epoch_start = time.time()

        running_loss = 0.0
        epoch_iterator = self.train_dataloader
        
        batches_cum_loss = 0
        batches_cum_time = 0
        batch_start = time.time()
        for i, batch in enumerate(epoch_iterator):  # Ignore the domain variable
            
            loss = self._batch_step(batch) #loss should be a result of a call to tensor.item()
        
            running_loss += loss
            batches_cum_loss += loss
            
            batch_end = time.time()
            batches_cum_time += (batch_end - batch_start)

            if self.use_wandb and (i+1)%self.log_every == 0:
                wandb.log({
                    'loss': batches_cum_loss / self.log_every,
                    'batch time': batches_cum_time / self.log_every,
                })
                batches_cum_loss = 0
                batches_cum_time = 0

            self._post_batch(i)

            batch_start = time.time()

        epoch_loss = running_loss / len(epoch_iterator)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        if self.use_wandb:
            wandb.log({
                'epoch_loss': epoch_loss, 
                'epoch time': epoch_time,
            })

        print(
                f"Epoch: {epoch} Epoch Loss: {epoch_loss} average batch time: {epoch_time / len(epoch_iterator)}"
        )

    def _post_batch(self, batch_num: int):
        return
    
    def _batch_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        assert len(batch) == 3
        inputs, labels, domains = batch
        inputs, labels, domains = inputs.to(self.device), labels.to(self.device), domains.to(self.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        loss = self._multi_domain_loss_function(outputs, labels, domains)
        assert loss.dim() == 0

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def _multi_domain_loss_function(self, preds: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        assert preds.shape[0] == labels.shape[0]

        return self.criterion(preds, labels)
    

    def _post_epoch(self, sets_to_eval: List[str], epoch: int, last_epoch: bool = False) -> None:
        self.model.eval()
        for dataset, dataloader in [
            ('train', self.train_dataloader), 
            ('in-domain-test', self.in_domain_test_dataloader),
            ('ood-test', self.ood_test_dataloader),
        ]:
            
            assert dataset in sets_to_eval
            pred_dict = self._predict_batched_data(dataloader)
            if (epoch == 4 or epoch == 6 or last_epoch) and self.use_wandb:
                self._log_preds(preds=pred_dict, data_set_name=dataset, epoch=epoch)
            eval_dict = self._eval(pred_dict)
            self._log_evaluation(eval_dict=eval_dict, label=dataset, log_metrics=True, log_table=False)    


    def _log_preds(self, preds, data_set_name, epoch):
        def log_key(key, arr):
            filename = f"{data_set_name}_{key}_epoch{epoch}__{run_unique_identifier}.npy"
            filepath = os.path.join(folder_path, filename)
            np.save(filepath, arr)
            preds_artifact.add_file(filepath)

        preds_artifact = wandb.Artifact(f'preds_{data_set_name}', type='dataset')
        run_unique_identifier = wandb.run.id
        folder_path = DUMP_FOLDER
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        for key, arr in preds.items():
            log_key(key, arr)

        # Log the artifact
        wandb.log_artifact(preds_artifact)

    def _predict_batched_data(self, dataloader: DataLoader, frac: float = 1.0) -> Dict[str, np.array]:
        self.model.eval()
        # self.model.train()
        labels_array = []
        logits_preds_array = []
        classes_preds_array = []
        domains_array = []

        # Convert dataloader to a list of batches
        batches = list(dataloader)

        # Determine the number of batches to sample (20% of the total)
        sample_size = int(len(batches) * frac)

        # Randomly sample 20% of batches
        sampled_batches = random.sample(batches, sample_size)

        # Now you can iterate over the sampled batches
        for batch in sampled_batches:
        # for batch in dataloader:
            inputs, labels, domains = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            classes_preds, logits_preds = self.model.predict_classes(inputs, return_logits=True)
            assert logits_preds.dim() == 2 and logits_preds.shape[1] == self.num_classes
            assert classes_preds.dim() == 2 and classes_preds.shape[1] == self.num_classes

            labels_array.extend(np.array(labels.cpu()))
            domains_array.extend(np.array(domains.cpu()))

            logits_preds_array.append(np.array(logits_preds.detach().cpu()))
            classes_preds_array.append(np.array(classes_preds.detach().cpu()))


        
        res_dict = {
            DataDumpTypes.LABEL: np.array(labels_array),
            DataDumpTypes.DOMAIN: np.array(domains_array),
            DataDumpTypes.LOGIT_PRED: np.vstack(logits_preds_array),
            DataDumpTypes.CLASS_PRED: np.vstack(classes_preds_array), 
        }

        assert res_dict[DataDumpTypes.LABEL].ndim == 1 and res_dict[DataDumpTypes.DOMAIN].ndim == 1 
        assert res_dict[DataDumpTypes.LOGIT_PRED].ndim == 2 and res_dict[DataDumpTypes.CLASS_PRED].ndim == 2
        assert res_dict[DataDumpTypes.LABEL].shape[0] == res_dict[DataDumpTypes.DOMAIN].shape[0]  == res_dict[DataDumpTypes.LOGIT_PRED].shape[0]  == res_dict[DataDumpTypes.CLASS_PRED].shape[0]
        assert res_dict[DataDumpTypes.LOGIT_PRED].shape[1] == res_dict[DataDumpTypes.CLASS_PRED].shape[1] == self.num_classes
        
        return res_dict
        
    
    def _eval(self, pred_dict: Dict[str, np.array]) -> Dict[str, Dict[str, float]]:
        correct = self._get_correct(pred_dict) #correct is np.array
        assert correct.ndim == 1 and correct.shape[0] == pred_dict[DataDumpTypes.LABEL].shape[0]

        preds_df = pd.DataFrame({
            DataDumpTypes.LABEL: pred_dict[DataDumpTypes.LABEL],
            DataDumpTypes.CORRECT: correct,
            DataDumpTypes.DOMAIN: pred_dict[DataDumpTypes.DOMAIN],
        })

        acc_per_domain = (
            preds_df
            .groupby(DataDumpTypes.DOMAIN)
            [DataDumpTypes.CORRECT]
            .mean()
            .to_frame()
            .rename(columns = {DataDumpTypes.CORRECT: 'acc'})
            .reset_index()
        )
        acc_per_domain_and_label = (
            preds_df
            .groupby([DataDumpTypes.DOMAIN, DataDumpTypes.LABEL])
            [DataDumpTypes.CORRECT]
            .mean()
            .unstack()
            .rename(columns = lambda s: 'recall_'+str(s))
            .reset_index()
        )
        
        eval_df =  (
            pd.merge(acc_per_domain, acc_per_domain_and_label, on=DataDumpTypes.DOMAIN)
            .set_index(DataDumpTypes.DOMAIN)
        )
        res = {col: {domain : eval_df.at[domain,col] for domain in eval_df.index} for col in eval_df.columns}
        return res
    

    def _get_correct(self, pred_dict: Dict[str, np.array]) -> np.array:
        predicted_classes = pred_dict[DataDumpTypes.CLASS_PRED]
        assert (
            isinstance(predicted_classes, np.ndarray) 
            and predicted_classes.ndim == 2 
            and predicted_classes.shape[0] == pred_dict[DataDumpTypes.LABEL].shape[0] 
            and predicted_classes.shape[1] == self.num_classes
        )
        true_class_idx = self._get_class_idx(pred_dict[DataDumpTypes.LABEL])
        return np.take_along_axis(predicted_classes, indices=true_class_idx.reshape(-1, 1), axis = 1).reshape(-1)


    def _get_class_idx(self, classes: np.array) -> np.array:
        assert isinstance(classes, np.ndarray) and classes.ndim == 1
        if self.classes_indeces_match_classes_values:
            return classes
        else:
            return np.array([self.classes[clss] for clss in classes])


    def _log_evaluation(
            self, 
            eval_dict: Dict[str, Dict[str, float]], 
            label: str, 
            log_metrics: bool = True, 
            log_table: bool = False
        ) -> None:
        
        if self.use_wandb and log_metrics:
            wandb.log({label: eval_dict})

        eval_df = (
            pd.DataFrame(eval_dict)
            .reset_index()
            .rename(columns={'index': 'domain'})
        )
        print(label + " eval: ")
        print(eval_df)
        print("================")
        print()
        if self.use_wandb and log_table:
            wandb.log({f"{label}_table": wandb.Table(dataframe=eval_df)})
    

class SetCoverTrainer(BasicNNTrainer):
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        criterion: Optional[torch.nn.Module], ## Creiterion will be ignored
        device: torch.device,
        train_dataset: MultiDomainsDataset, 
        in_domain_test_dataset: MultiDomainsDataset, 
        ood_test_dataset: MultiDomainsDataset, 
        batch_size: int = 64,
        num_epochs: int = 10, 
        use_wandb: bool = False,
        log_every: int = 10,    
        coverage: float = 0.9, 
        C: float = 0.5,
        update_c_every: int = 500,
        use_alternative_size_loss: bool = False,
    ):
        super().__init__(
            model=model, 
            optimizer=optimizer,
            criterion=None,
            device=device,
            train_dataset=train_dataset, 
            in_domain_test_dataset=in_domain_test_dataset, 
            ood_test_dataset=ood_test_dataset, 
            batch_size=batch_size,
            num_epochs=num_epochs, 
            use_wandb=use_wandb,
            log_every=log_every,
        )
        
        self.coverage = coverage
        self.default_C = C
        self.update_c_every = update_c_every
        self.use_alternative_size_loss = use_alternative_size_loss

        self.C = self.default_C * torch.ones(self.num_classes, self.num_domains, requires_grad=False)
        self.C = self.C.to(self.device)


    def _create_dataloaders(self, batch_size: int) -> None:        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.in_domain_test_dataloader = DataLoader(self.in_domain_test_dataset, batch_size=batch_size, shuffle=True)
        self.ood_test_dataloader =  DataLoader(self.ood_test_dataset, batch_size=batch_size, shuffle=True)

    
    def _multi_domain_loss_function(self, preds: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        assert preds.dim() == 2
        assert labels.dim() == 1
        assert preds.shape[0] == labels.shape[0]

        loss = torch.tensor(0, dtype=torch.float32, device = self.device)
        for clss in self.classes.keys():
            cls_idx = self.classes[clss]
            
            y_pred_cls = preds[:, cls_idx]
            assert y_pred_cls.dim() == 1 and y_pred_cls.shape[0] == labels.shape[0]
            hinge_loss_objective = torch.nn.MarginRankingLoss(margin = 1, reduction = 'none')
            hinge_loss_constraints = torch.nn.MarginRankingLoss(margin=1, reduction='none')
            constraint_loss_per_i = hinge_loss_constraints(y_pred_cls, torch.zeros_like(y_pred_cls), torch.ones_like(y_pred_cls))
            set_size_loss_per_i = hinge_loss_objective(y_pred_cls, torch.zeros_like(y_pred_cls), -1*torch.ones_like(y_pred_cls))

            alternative_set_size_condition = ~(labels == clss)
            if self.use_alternative_size_loss:
                set_size_condition = alternative_set_size_condition
            else:
                set_size_condition = ~alternative_set_size_condition | alternative_set_size_condition
            domain_loss_condition = (labels == clss)
            # Map domains to their indices using list comprehension
            domain_indices = torch.tensor([self.domains[domain.item()] for domain in domains])
            # Use the domain indices to select the corresponding rows from C
            C_values = torch.nan_to_num(self.C[cls_idx][domain_indices])

            cls_loss = torch.mean(
                set_size_condition * set_size_loss_per_i 
                + 
                C_values * constraint_loss_per_i * domain_loss_condition
            )

            loss += cls_loss

        return loss

    
    def _post_batch(self, batch_num: int):
        # return
        if (batch_num+1) % self.update_c_every == 0:
            self.model.eval()
            pred_dict = self._predict_batched_data(self.train_dataloader, frac=0.5)
            eval_dict = self._eval(pred_dict)
            self._update_C_param(eval_dict)
            self.model.train()
        return
    
    def _post_epoch(self, sets_to_eval: List[str], epoch: int, last_epoch: bool = False) -> None:
        self.model.eval()
        for dataset, dataloader in [
            ('train', self.train_dataloader), 
            ('in-domain-test', self.in_domain_test_dataloader),
            ('ood-test', self.ood_test_dataloader),
        ]:
            
            assert dataset in sets_to_eval
            pred_dict = self._predict_batched_data(dataloader)
            if (epoch==4 or epoch==6 or last_epoch) and self.use_wandb:
                self._log_preds(preds=pred_dict, data_set_name=dataset, epoch=epoch)

            eval_dict = self._eval(pred_dict)
            if dataset == "train":
                self._update_C_param(eval_dict)
            self._log_evaluation(eval_dict=eval_dict, label=dataset, log_metrics=True, log_table=False)
            
    

    def _eval(self, pred_dict: Dict[str, np.array]) -> Dict[str, Dict[str, float]]:
        acc_recall_dict = super()._eval(pred_dict)

        assert pred_dict[DataDumpTypes.CLASS_PRED].ndim == 2
        assert pred_dict[DataDumpTypes.CLASS_PRED].shape[0] == pred_dict[DataDumpTypes.DOMAIN].shape[0]
        size = (pred_dict[DataDumpTypes.CLASS_PRED] == 1).sum(axis = 1)
        assert size.shape[0] == pred_dict[DataDumpTypes.DOMAIN].shape[0]
        preds_df = pd.DataFrame({
            'size': size,
            DataDumpTypes.DOMAIN: pred_dict[DataDumpTypes.DOMAIN],
        })

        mean_size_df = preds_df.groupby(DataDumpTypes.DOMAIN)['size'].mean().to_frame()        
        size_dict = {col: {domain : mean_size_df.at[domain,col] for domain in mean_size_df.index} for col in mean_size_df.columns}
        return {**acc_recall_dict, **size_dict}


    def _update_C_param(self, train_eval_dict: Dict[str, Dict[str, float]]) -> None:
        assert all([metric.split('_')[0] in ['acc', 'recall', 'size'] for metric in train_eval_dict.keys()])
        factors = []
        for metric, domains_dict in train_eval_dict.items():
            metric_split = metric.split('_')
            if not metric_split[0] == 'recall':
                continue
            clss = metric_split[1]
            cls_idx = self.classes[float(clss)] ## keys of self.classes are float or int

            for domain, cls_domain_recall in domains_dict.items():
                domain_idx = self.domains[float(domain)] ## keys of self.domains are float or int
                update_factor = 1 - (cls_domain_recall - self.coverage)
                factors.append(update_factor)
                self.C[cls_idx][domain_idx] *= (
                    update_factor * (2 if update_factor > 1 else 1)
                )
            
        print(self.C)
        mean_size = np.mean([size for domain, size in train_eval_dict['size'].items()])
        print(
            f"mean set size: {mean_size}"
            f"mean factor: {1 - np.mean(factors) + self.coverage}, "
            f"min factor : {1 - np.max(factors) + self.coverage}, "
            f"max C: {torch.max(self.C)}"
        )


class DomainBedTrainer(BasicNNTrainer):
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        criterion: Optional[torch.nn.Module], ## Creiterion will be ignored
        device: torch.device,
        train_dataset: MultiDomainsDataset, 
        in_domain_test_dataset: MultiDomainsDataset, 
        ood_test_dataset: MultiDomainsDataset, 
        batch_size: int = 64,
        num_epochs: int = 10, 
        use_wandb: bool = False,
        log_every: int = 10,    
    ):
        super().__init__(
            model=model, 
            optimizer=optimizer,
            criterion=None,
            device=device,
            train_dataset=train_dataset, 
            in_domain_test_dataset=in_domain_test_dataset, 
            ood_test_dataset=ood_test_dataset, 
            batch_size=batch_size,
            num_epochs=num_epochs, 
            use_wandb=use_wandb,
            log_every=log_every,
        )


    def _create_dataloaders(self, batch_size: int) -> None:
        train_sampler = CustomBatchSampler(
            domains=self.train_dataset.domains, 
            mini_batch_size = batch_size//4, 
            domains_per_batch = 4
        )
        self.train_dataloader = DataLoader(self.train_dataset, batch_sampler = train_sampler)

        in_domain_test_sampler = CustomBatchSampler(
            domains=self.in_domain_test_dataset.domains, 
            mini_batch_size = batch_size//4, 
            domains_per_batch = 4
        )
        self.in_domain_test_dataloader = DataLoader(self.in_domain_test_dataset, batch_sampler = in_domain_test_sampler)
        
        ood_test_sampler = CustomBatchSampler(
            domains=self.ood_test_dataset.domains, 
            mini_batch_size = batch_size//4, 
            domains_per_batch = 4
        )
        self.ood_test_dataloader = DataLoader(self.ood_test_dataset, batch_sampler = ood_test_sampler)


    def _batch_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        assert len(batch) == 3
        inputs, labels, domains = batch
        inputs, labels, domains = inputs.to(self.device), labels.to(self.device), domains.to(self.device)
        unique_domains = torch.unique(domains)
        minibatches = []
        for domain in unique_domains:
            domain_mask = domains == domain
            minibatch_x = inputs[domain_mask]
            minibatch_y = labels[domain_mask]
            minibatches.append((minibatch_x, minibatch_y))
        
        return self.model.update(minibatches)