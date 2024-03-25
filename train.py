import torch
import wandb
from utils import (
    parse_args, 
    set_env_variables, 
    set_seed, 
    log_args_to_wandb, 
    log_data,
    ModelTypeFactory,
)
from experiments.nn_trainers import BasicNNTrainer
from experiments.utils import DUMP_FOLDER
import time


def get_trainer(args, device: str) -> BasicNNTrainer:
    model_type_factory = ModelTypeFactory(
        input_model_type=args.model_type, 
        base_model=args.base_model, 
        dataset=args.dataset,
    )
    model, transform = model_type_factory.get_model_and_transform()

    print("getting data")
    train_subset, in_domain_test_subset, ood_test_subset = model_type_factory.get_train_test_datasets(
        dataset = args.dataset,
        in_domain_domains = args.train_domains, 
        ood_domains = args.ood_domains, 
        num_train_domains = args.num_train_domains,
        num_ood_domains = args.num_ood_domains,
        train_domain_size = args.train_domain_size, 
        test_domain_size = args.test_domain_size, 
        use_preloaded_dataset = args.use_preloaded_dataset,
        transform=transform,
        min_domain_label_cnt = args.min_domain_label_cnt,
        CVC_split = args.CVC_split,
    )

    if args.use_wandb:
        log_data(train_subset, in_domain_test_subset, ood_test_subset)

    trainer = model_type_factory.get_trainer(
        model=model, 
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        train_dataset=train_subset, 
        in_domain_test_dataset=in_domain_test_subset, 
        ood_test_dataset=ood_test_subset, 
        batch_size=args.batch_size,
        num_epochs=args.epochs, 
        use_wandb=args.use_wandb,
        log_every=args.batch_log_freq,
        coverage=args.coverage, 
        C=args.C,
        update_c_every=args.update_c_every,
        use_alternative_size_loss=args.use_alternative_size_loss,
    )

    return trainer


def main():
    print('starting script')
    args = parse_args()

    set_env_variables()
    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        log_args_to_wandb(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Should be using device {device}")

    trainer = get_trainer(args, device)

    trainer.train_model()

    if DUMP_FOLDER:
        run_unique_identifier = wandb.run.id if args.use_wandb else time.strftime("%Y%m%d-%H%M%S")
        file_name = args.model_type + '__' + args.base_model + '__' + run_unique_identifier
        folder_name = DUMP_FOLDER
        trainer.save_model(folder_path = folder_name, filename = file_name)


if __name__ == "__main__":
    main()
