#!/bin/bash
set -e

PATH_TO_DATA_DIR="data"

CORD_CONFIG="--dataset-name cord --data_dir $PATH_TO_DATA_DIR/cord/"
FUNSD_CONFIG="--dataset-name funsd --data_dir $PATH_TO_DATA_DIR/funsd/"
SROIE_CONFIG="--dataset-name sroie --data_dir $PATH_TO_DATA_DIR/sroie/"
WILD_RECEIPTS_CONFIG="--dataset-name wild_receipts --data_dir $PATH_TO_DATA_DIR/wild_receipts/"

BERT_ARGS="--model-name bert-base-uncased --builder-type atria --tokenizer-name bert-base-uncased"
ROBERTA_ARGS="--model-name roberta-base --builder-type atria --tokenizer-name roberta-base"
LILT_ROBERTA_ARGS="--model-name lilt-roberta-base --builder-type atria --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base"
LAYOUTLMV3_ARGS="--model-name layoutlmv3-base --builder-type atria --tokenizer-name microsoft/layoutlmv3-base --use_segment_level_bboxes True"

declare -a eval_args=(
    # cord configs
    "$CORD_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_ser/cord/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$CORD_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/cord/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$CORD_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/cord/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$CORD_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_ser/cord/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"

    # funsd configs
    "$FUNSD_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_ser/funsd/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$FUNSD_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/funsd/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$FUNSD_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/funsd/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$FUNSD_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_ser/funsd/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"

    # wild_receipts configs
    "$WILD_RECEIPTS_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_ser/wild_receipts/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$WILD_RECEIPTS_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/wild_receipts/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$WILD_RECEIPTS_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/wild_receipts/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$WILD_RECEIPTS_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_ser/wild_receipts/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"

    # sroie configs
    "$SROIE_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_ser/sroie/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$SROIE_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/sroie/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$SROIE_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_ser/sroie/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$SROIE_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_ser/sroie/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"
)

EVAL_BATCH_SIZE=1
GRAD_INTERNAL_BATCH_SIZE=6
PERT_INTERNAL_BATCH_SIZE=8

declare explainer_args=(
    "--explainer-name grad/saliency --eval-batch-size $EVAL_BATCH_SIZE --grad_batch_size 16"
    "--explainer-name grad/deeplift --eval-batch-size $EVAL_BATCH_SIZE --grad_batch_size 16"
    "--explainer-name grad/input_x_gradient --eval-batch-size $EVAL_BATCH_SIZE --grad_batch_size 16"
    "--explainer-name grad/guided_backprop --eval-batch-size $EVAL_BATCH_SIZE --grad_batch_size 16"
    "--explainer-name grad/gradient_shap --eval-batch-size 1 --internal_batch_size $GRAD_INTERNAL_BATCH_SIZE --grad_batch_size 16"
    "--explainer-name grad/integrated_gradients --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size 16 --grad_batch_size 1"
    "--explainer-name grad/deeplift_shap --eval-batch-size 1 --internal_batch_size 16 --grad_batch_size 1"
    "--explainer-name perturbation/feature_ablation --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/lime --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/kernel_shap --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/occlusion --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
)

PROJECT_NAME="docxeval"
EXP_NAME="experiment_02_ser"
TASK="token_classification"
COMMON_ARGS="--total_samples 100 --compute-metrics True --only-load-cached-explanations False" # this is the max total samples

declare -a CONFIGS=()
for eval_arg in "${eval_args[@]}"; do
    DATASET_NAME=$(sed -n 's/.*--dataset-name \([^ ]*\).*/\1/p' <<< "$eval_arg")
    DATASET_NAME=${DATASET_NAME//\//_}  # replace '/' with '_'
    MODEL_NAME=$(sed -n 's/.*--model-name \([^ ]*\).*/\1/p' <<< "$eval_arg") 

    for explainer_arg in "${explainer_args[@]}"; do
        EXPLAINER_NAME=$(sed -n 's/.*--explainer-name \([^ ]*\).*/\1/p' <<< "$explainer_arg")
        EXPLAINER_NAME=${EXPLAINER_NAME// /_}  # replace ' ' with '_'
        job_name="${EXP_NAME}_${DATASET_NAME}_${MODEL_NAME}_${EXPLAINER_NAME}"
        script="python src/docxeval/explainers/${TASK}.py --project-name "$PROJECT_NAME" --exp-name "$EXP_NAME" --output-dir ../outputs/ $eval_arg $explainer_arg $COMMON_ARGS"
        CONFIGS+=("$job_name $script")
    done 
done