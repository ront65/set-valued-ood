#!/bin/bash


#SBATCH --job-name=ront_train_resnet
#SBATCH --output=/vol/scratch/rontsibulsky/slurm_outputs/slurm_%j.out # redirect stdout
#SBATCH --error=/vol/scratch/rontsibulsky/slurm_outputs/slurm_%j.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)


# Default values
use_wandb=""
wandb_project=""
wandb_entity=""
train_patients=()
ood_patients=()
num_train_domains=""
num_ood_domains=""
train_domain_size=""
test_domain_size=""
use_preloaded_dataset=""
cancel_preloaded_dataset=""
batch_log_freq=""
seed=""
model_type=""
base_model=""
dataset=""
CVC_split=""

lr=""
batch_size=""
epochs=""
min_domain_label_cnt=""
coverage=""
C=""
update_c_every=""
use_alternative_size_loss=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --use-wandb) use_wandb="--use-wandb"; shift ;;
        --wandb-project) wandb_project="$2"; shift 2;;
        --wandb-entity) wandb_entity="$2"; shift 2;;
        --train-patients) shift; while [[ "$1" && ! "$1" =~ ^-- ]]; do train_patients+=("$1"); shift; done ;;
        --ood-patients) shift; while [[ "$1" && ! "$1" =~ ^-- ]]; do ood_patients+=("$1"); shift; done ;;
        --num-train-domains) num_train_domains="$2"; shift 2 ;;
        --num-ood-domains) num_ood_domains="$2"; shift 2 ;;
        --train-domain-size) train_domain_size="$2"; shift 2 ;;
        --test-domain-size) test_domain_size="$2"; shift 2 ;;
        --use-preloaded-dataset) use_preloaded_dataset="--use-preloaded-dataset"; shift ;;
        --cancel-preloaded_dataset) cancel_preloaded_dataset="--cancel-preloaded_dataset"; shift ;;
        --batch-log-freq) batch_log_freq="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --model-type) model_type="$2"; shift 2 ;;
        --base-model) base_model="$2"; shift 2 ;;
        --dataset) dataset="$2"; shift 2 ;;
        --CVC-split) CVC_split="--CVC-split"; shift ;;

        --lr) lr="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --min-domain-label-cnt) min_domain_label_cnt="$2"; shift 2 ;;
        --coverage) coverage="$2"; shift 2 ;;
        --C) C="$2"; shift 2 ;;
        --update-c-every) update_c_every="$2"; shift 2 ;;
        --use-alternative-size-loss) use_alternative_size_loss="--use-alternative-size-loss"; shift ;;


        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Build the command
cmd="python /vol/scratch/rontsibulsky/test/set-valued-domain-generalization/camelyon/train.py"

if [[ -n "$use_wandb" ]]; then cmd+=" $use_wandb"; fi
if [[ -n "$wandb_project" ]]; then cmd+=" --wandb-project $wandb_project"; fi
if [[ -n "$wandb_entity" ]]; then cmd+=" --wandb-entity $wandb_entity"; fi
if [[ ${#train_patients[@]} -gt 0 ]]; then cmd+=" --train-patients ${train_patients[@]}"; fi
if [[ ${#ood_patients[@]} -gt 0 ]]; then cmd+=" --ood-patients ${ood_patients[@]}"; fi
if [[ -n "$num_train_domains" ]]; then cmd+=" --num-train-domains $num_train_domains"; fi
if [[ -n "$num_ood_domains" ]]; then cmd+=" --num-ood-domains $num_ood_domains"; fi
if [[ -n "$train_domain_size" ]]; then cmd+=" --train-domain-size $train_domain_size"; fi
if [[ -n "$test_domain_size" ]]; then cmd+=" --test-domain-size $test_domain_size"; fi
if [[ -n "$use_preloaded_dataset" ]]; then cmd+=" $use_preloaded_dataset"; fi
if [[ -n "$cancel_preloaded_dataset" ]]; then cmd+=" $cancel_preloaded_dataset"; fi
if [[ -n "$batch_log_freq" ]]; then cmd+=" --batch-log-freq $batch_log_freq"; fi
if [[ -n "$seed" ]]; then cmd+=" --seed $seed"; fi
if [[ -n "$model_type" ]]; then cmd+=" --model-type $model_type"; fi
if [[ -n "$base_model" ]]; then cmd+=" --base-model $base_model"; fi
if [[ -n "$dataset" ]]; then cmd+=" --dataset $dataset"; fi
if [[ -n "$CVC_split" ]]; then cmd+=" $CVC_split"; fi

if [[ -n "$lr" ]]; then cmd+=" --lr $lr"; fi
if [[ -n "$batch_size" ]]; then cmd+=" --batch-size $batch_size"; fi
if [[ -n "$epochs" ]]; then cmd+=" --epochs $epochs"; fi
if [[ -n "$min_domain_label_cnt" ]]; then cmd+=" --min-domain-label-cnt $min_domain_label_cnt"; fi
if [[ -n "$coverage" ]]; then cmd+=" --coverage $coverage"; fi
if [[ -n "$C" ]]; then cmd+=" --C $C"; fi
if [[ -n "$update_c_every" ]]; then cmd+=" --update-c-every $update_c_every"; fi
if [[ -n "$use_alternative_size_loss" ]]; then cmd+=" $use_alternative_size_loss"; fi




# Run the command
echo "Running command: $cmd"
eval "$cmd"