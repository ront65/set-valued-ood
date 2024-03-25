import numpy as np
import random
import os
import argparse
import torch
from torchvision import models
import wandb
from enum import Enum
from experiments.nn_trainers import BasicNNTrainer, SetCoverTrainer, DomainBedTrainer
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds import get_dataset
from Data.wilds.wilds_data import get_wilds_train_test_subsets
from Data.wilds.utils import (
    iwildcam_preprocess, 
    camelyon_preprocess, 
    fmow_preprocess,
    amazon_preprocess,
    IWILDCAM_FILTERED_LABELS,
    FMOW_FILTERED_LABELS
)
from Data.synthetic.synthetic_data import (
    get_synthetic_train_test_subsets, 
    SYNTHETIC_2_PARAMS,
    SYNTHETIC_10_PARAMS,
    SYNTHETIC_50_PARAMS,
)

from models.utils import (
    PositiveOutputsAsClassPreds,
    MaxOutputAsClassPred,
    ModelWithAddedSoftmax,
    DomainBedWrapper,
) 
from models.custom_mlp import CustomMLP
from domainbed.algorithms import IRM, VREx, CORAL, MMD
from collections import namedtuple
from experiments.utils import DUMP_FOLDER

class ModelTypes(Enum):
    STANDARD = "standard"
    SET_COVER = "set_cover"
    DOMAIN_BED = "domain_bed"

class DataSets(Enum):
    CAMELYON = "camelyon"
    IWILDCAM = "iwildcam"
    SYNTHETIC_2 = "synthetic_2"
    SYNTHETIC_10 = "synthetic_10"
    SYNTHETIC_50 = "synthetic_50"
    FMOW = "fmow"
    AMAZON = "amazon"

WildsDataSetParams = namedtuple('WildsDataSetParams', ['name', 'preprocess_func'])
WILDS_PARAMS_DICT={
    DataSets.CAMELYON: WildsDataSetParams('camelyon17', camelyon_preprocess),
    DataSets.IWILDCAM: WildsDataSetParams('iwildcam', iwildcam_preprocess),
    DataSets.FMOW: WildsDataSetParams('fmow', fmow_preprocess),
    DataSets.AMAZON: WildsDataSetParams('amazon', amazon_preprocess),
}

ModelDimensions = namedtuple('ModelDimensions', ['input', 'hidden', 'output'])
MODEL_DIM_DICT={
    ## Image datasets are trained with Resnet and only need to set output dim
    DataSets.CAMELYON: ModelDimensions(None, None, 2),
    DataSets.IWILDCAM: ModelDimensions(None, None, len(IWILDCAM_FILTERED_LABELS)),
    DataSets.FMOW: ModelDimensions(None, None, len(FMOW_FILTERED_LABELS)),
    ##Datasets trained with MLP need to set all dims
    DataSets.AMAZON: ModelDimensions(768, 10, 5),
    DataSets.SYNTHETIC_2: ModelDimensions(2, 10, 2),
    DataSets.SYNTHETIC_10: ModelDimensions(10, 20, 2),
    DataSets.SYNTHETIC_50: ModelDimensions(50, 20, 2),
}
#TODO: comment out domained
class BaseModel(Enum):
    RESNET = "resnet"
    MLP = "mlp"
    IRM = "irm" #IRM uses resnet18
    VREx = "vrex"
    CORAL = "coral"
    MMD = "mmd"


IRM_hparams = {
    'nonlinear_classifier': False, 
    'lr': 0.001, 
    'weight_decay': 0., 
    'vit': False, 
    'resnet18': True, 
    'resnet50_augmix': False, 
    'freeze_bn': False, 
    'resnet_dropout': 0, 
    'irm_lambda':100 , 
    'irm_penalty_anneal_iters': 500
}

VREx_hparams = {
    'nonlinear_classifier': False, 
    'lr': 0.001, 
    'weight_decay': 0., 
    'vit': False, 
    'resnet18': True, 
    'resnet50_augmix': False, 
    'freeze_bn': False, 
    'resnet_dropout': 0, 
    'vrex_lambda': 10,
    'vrex_penalty_anneal_iters': 500,
}

CORAL_hparams = {
    'nonlinear_classifier': False, 
    'lr': 0.001, 
    'weight_decay': 0., 
    'vit': False, 
    'resnet18': True, 
    'resnet50_augmix': False, 
    'freeze_bn': False, 
    'resnet_dropout': 0, 
    'mmd_gamma': 1.
}

MMD_hparams = {
    'nonlinear_classifier': False, 
    'lr': 0.001, 
    'weight_decay': 0., 
    'vit': False, 
    'resnet18': True, 
    'resnet50_augmix': False, 
    'freeze_bn': False, 
    'resnet_dropout': 0, 
    'mmd_gamma': 1.
}

class ModelTypeFactory:

    def __init__(self, input_model_type, base_model, dataset):
        assert input_model_type in [allowd_type.value for allowd_type in ModelTypes]
        assert base_model in [allowd_type.value for allowd_type in BaseModel]
        assert dataset in [allowd_type.value for allowd_type in DataSets]

        self.model_type = ModelTypes(input_model_type)
        self.base_model = BaseModel(base_model)
        self.dataset = DataSets(dataset)

    def get_train_test_datasets(self, **kwargs):
        ## Wilds dataset
        if self.dataset in WILDS_PARAMS_DICT.keys():
            wilds_dataset_params = WILDS_PARAMS_DICT[self.dataset]
            dataset = get_dataset(dataset=wilds_dataset_params.name, download=True)
            dataset_preprocessed = wilds_dataset_params.preprocess_func(
                dataset, 
                min_domain_label_cnt=kwargs['min_domain_label_cnt']
            )
            return get_wilds_train_test_subsets(
                main_data_set=dataset_preprocessed, 
                **kwargs
            )
        ## Synthetic dataset
        elif self.dataset == DataSets.SYNTHETIC_2:
            return get_synthetic_train_test_subsets(data_params = SYNTHETIC_2_PARAMS, allow_log_data=True, **kwargs)
        
        elif self.dataset == DataSets.SYNTHETIC_10:
            return get_synthetic_train_test_subsets(data_params = SYNTHETIC_10_PARAMS, **kwargs)
        
        elif self.dataset == DataSets.SYNTHETIC_50:
            return get_synthetic_train_test_subsets(data_params = SYNTHETIC_50_PARAMS, **kwargs)

        else:
            raise Exception(f"Should not get to this line of code - somethin is wrong with given dataset {dataset}")


    def get_model_and_transform(self,):
        ##RESNET
        if not self.dataset in MODEL_DIM_DICT.keys():
            raise Exception(f"Got unsupported dataset {self.dataset} with base model {self.base_model}")
        model_dims = MODEL_DIM_DICT[self.dataset]
        if self.base_model == BaseModel.RESNET:
            model, transform = get_resnet_model_and_transform(num_out_features=model_dims.output)
            
        ##IRM
        elif self.base_model == BaseModel.IRM:
            model, transform = get_irm_model_and_transform(num_out_features=model_dims.output)
        
        ##VREx
        elif self.base_model == BaseModel.VREx:
            model, transform = get_vrex_model_and_transform(num_out_features=model_dims.output)

        ##CORAL
        elif self.base_model == BaseModel.CORAL:
            model, transform = get_coral_model_and_transform(num_out_features=model_dims.output)

        ##MMD
        elif self.base_model == BaseModel.MMD:
            model, transform = get_mmd_model_and_transform(num_out_features=model_dims.output)

        ##MLP
        elif self.base_model == BaseModel.MLP:
            model, transform = get_mlp_model_and_transform(
                input_dim=model_dims.input, 
                hidden_dim=model_dims.hidden,
                output_dim=model_dims.output
            )

        else:
            raise Exception(f"Should not get to this line of code - somethin is wrong with given base model {self.base_model}")


        if self.model_type == ModelTypes.SET_COVER:
            model = PositiveOutputsAsClassPreds(model)

        elif self.model_type == ModelTypes.STANDARD:
            model = MaxOutputAsClassPred(ModelWithAddedSoftmax(model))
        
        elif self.model_type == ModelTypes.DOMAIN_BED:
            model = DomainBedWrapper(model)

        else:
            raise Exception("Should not get to this line of code - somethin is wrong with given model type")

        return model, transform
    

    def get_trainer(
        self,
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        train_dataset: WILDSDataset, 
        in_domain_test_dataset: WILDSDataset, 
        ood_test_dataset: WILDSDataset, 
        batch_size: int = 64,
        num_epochs: int = 10, 
        use_wandb: bool = False,
        log_every: int = 10,    
        coverage: float = 0.9, 
        C: float = 0.5,
        update_c_every: int = 500,
        use_alternative_size_loss: bool = False,

    ) -> BasicNNTrainer:

        if self.model_type == ModelTypes.STANDARD:
            trainer = BasicNNTrainer(
                model=model, 
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                train_dataset=train_dataset, 
                in_domain_test_dataset=in_domain_test_dataset, 
                ood_test_dataset=ood_test_dataset, 
                batch_size=batch_size,
                num_epochs=num_epochs, 
                use_wandb=use_wandb,
                log_every=log_every,
            )

        elif self.model_type == ModelTypes.SET_COVER:
            trainer = SetCoverTrainer(
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
                coverage = coverage, 
                C=C,
                update_c_every=update_c_every,
                use_alternative_size_loss=use_alternative_size_loss,
            )

        elif self.model_type == ModelTypes.DOMAIN_BED:
            trainer = DomainBedTrainer(
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

        else:
            raise Exception("Should not get to this line of code - somethin is wrong with given model type")
            
        return trainer

#TODO: Delete
def set_env_variables():
    os.environ['TORCH_HOME'] = '/vol/scratch/rontsibulsky/torch_cache'
    os.environ['WANDB_DIR'] = '/vol/scratch/rontsibulsky/wandb_data'  # Set your desired directory her


def get_resnet_model_and_transform(num_out_features: int):
    # Load the pretrained ResNet18 model
    model = models.resnet18(weights='DEFAULT')
    
    # Modify the final layer to output binary class probabilities
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_out_features)  # Binary classification

    transform = models.ResNet18_Weights.DEFAULT.transforms()
    
    return model, transform


def get_irm_model_and_transform(num_out_features: int):
    model = IRM(input_shape=(3, 224, 224), num_classes=num_out_features, num_domains=None, hparams=IRM_hparams)    
    transform = models.ResNet18_Weights.DEFAULT.transforms()
    
    return model, transform


def get_vrex_model_and_transform(num_out_features: int):
    model = VREx(input_shape=(3, 224, 224), num_classes=num_out_features, num_domains=None, hparams=VREx_hparams)    
    transform = models.ResNet18_Weights.DEFAULT.transforms()
    
    return model, transform


def get_coral_model_and_transform(num_out_features: int):
    model = CORAL(input_shape=(3, 224, 224), num_classes=num_out_features, num_domains=None, hparams=CORAL_hparams)    
    transform = models.ResNet18_Weights.DEFAULT.transforms()
    
    return model, transform


def get_mmd_model_and_transform(num_out_features: int):
    model = MMD(input_shape=(3, 224, 224), num_classes=num_out_features, num_domains=None, hparams=MMD_hparams)    
    transform = models.ResNet18_Weights.DEFAULT.transforms()
    
    return model, transform


def get_mlp_model_and_transform(input_dim: int, hidden_dim: int, output_dim: int = 2):
    model = CustomMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    transform = None

    return model, transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument('--use-wandb', action='store_true', help='Use WANDB for logging')
    parser.add_argument('--wandb-project', default='Camelyon', help='Name of Wandb project to use')
    parser.add_argument('--wandb-entity', default='', help='Name of Wandb entity to use')
    parser.add_argument('--train-domains', nargs='*', help='List of training domains, default is None')
    parser.add_argument('--ood-domains', nargs='*', default=None, help='List of OOD domains, default is None')
    parser.add_argument('--num-train-domains', type=int, default=None, help='if train patients are not specified, sample randomly. This is the fraction / number of train patients to sample.')
    parser.add_argument('--num-ood-domains', type=int, default=None, help='if train patients are not specified, sample randomly. This is the fraction / number of ood patients to sample.')
    parser.add_argument('--train-domain-size', type=int, default=2000, help='Size of training domain')
    parser.add_argument('--test-domain-size', type=int, default=2000, help='Size of test domain')
    parser.add_argument('--use-preloaded-dataset', action='store_true', help='Use preloaded dataset (default: True)')
    parser.add_argument('--cancel_preloaded_dataset', action='store_false', dest='use_preloaded_dataset', help='Do not use preloaded dataset')
    parser.set_defaults(use_preloaded_dataset=True)
    parser.add_argument('--batch-log-freq', type=int, default=10, help='Batch log frequency')
    parser.add_argument('--seed', type=int, default=123, help='Seed for randomness reproduciility')
    parser.add_argument('--model-type', required=True, choices = [model_type.value for model_type in ModelTypes], help='Type of model to use')
    parser.add_argument('--base-model', required=True, choices = [model_type.value for model_type in BaseModel], help='Type of base model to use')
    parser.add_argument('--dataset', required=True, choices = [model_type.value for model_type in DataSets], help='Type of base model to use')
    parser.add_argument('--CVC-split', action='store_true', help='Split train data into validation domains (moved to ood group) and train domains')
    

    #Training params
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--min-domain-label-cnt', type=int, default=100, help='Min number of data sample for each pair of domain X label')
    parser.add_argument('--coverage', type=float, default=0.9, help='target coverage for set-valued predictions')
    parser.add_argument('--C', type=float, default=5.0, help='Starting C values')
    parser.add_argument('--update-c-every', type=int, default=500, help='Update C values once every __ batches, in addition to update after each epoch')
    parser.add_argument('--use-alternative-size-loss', action='store_true', help='Use set size minus true label loss, insteead of regular set size loss')
    return parser.parse_args()


def log_args_to_wandb(args):
    args_dict = vars(args)  
    wandb.config.update(args_dict)  


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    os.environ['PYTHONHASHSEED'] = str(seed)


def log_data(train_subset, in_domain_test_subset, ood_test_subset):
    if not DUMP_FOLDER:
        return
    run_unique_identifier = wandb.run.id
    folder_path = f'{DUMP_FOLDER}/{run_unique_identifier}/'
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    data_artifact = wandb.Artifact('data', type='dataset')
    # Save the data to the artifact

    train_data_file_path = train_subset.log_data('train', folder_path)
    if train_data_file_path is not None:
        data_artifact.add_file(train_data_file_path)

    in_domain_test_data_file_path = in_domain_test_subset.log_data('in_domain_test', folder_path)
    if in_domain_test_data_file_path is not None:
        data_artifact.add_file(in_domain_test_data_file_path)
    
    if ood_test_subset is not None:
        ood_test_data_file_path = ood_test_subset.log_data('ood_test', folder_path)
        if ood_test_data_file_path is not None:
            data_artifact.add_file(ood_test_data_file_path)
    
    # Log the artifact
    wandb.log_artifact(data_artifact)