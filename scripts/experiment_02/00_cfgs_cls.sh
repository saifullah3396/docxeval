#!/bin/bash
set -e

PATH_TO_DATA_DIR="data"

TOBACCO3482_CONFIG="--dataset-name tobacco3482/image_with_ocr --data-dir $PATH_TO_DATA_DIR/tobacco3482/"
RVLCDIP_CONFIG="--dataset-name rvlcdip/image_with_ocr --data-dir $PATH_TO_DATA_DIR/rvlcdip/"
DOCLAYNET_CONFIG="--dataset-name doclaynet/default --data-dir $PATH_TO_DATA_DIR/doclaynet"

BERT_ARGS="--model-name bert-base-uncased --builder-type atria --tokenizer-name bert-base-uncased"
ROBERTA_ARGS="--model-name roberta-base --builder-type atria --tokenizer-name roberta-base"
LILT_ROBERTA_ARGS="--model-name lilt-roberta-base --builder-type atria --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base"
LAYOUTLMV3_ARGS="--model-name layoutlmv3-base --builder-type atria --tokenizer-name microsoft/layoutlmv3-base"

declare -a eval_args=(
    # tobacco3482 configs
    "$TOBACCO3482_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_cls/tobacco3482_image_with_ocr/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$TOBACCO3482_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_cls/tobacco3482_image_with_ocr/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$TOBACCO3482_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_cls/tobacco3482_image_with_ocr/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$TOBACCO3482_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_cls/tobacco3482_image_with_ocr/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"

    # rvlcdip configs
    "$RVLCDIP_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_cls/rvlcdip_image_with_ocr/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$RVLCDIP_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_cls/rvlcdip_image_with_ocr/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$RVLCDIP_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_cls/rvlcdip_image_with_ocr/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$RVLCDIP_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_cls/rvlcdip_image_with_ocr/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"

    # doclaynet configs
    "$DOCLAYNET_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_cls/doclaynet_default/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$DOCLAYNET_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_cls/doclaynet_default/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$DOCLAYNET_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_cls/doclaynet_default/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$DOCLAYNET_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_cls/doclaynet_default/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"
)

EVAL_BATCH_SIZE=1
GRAD_INTERNAL_BATCH_SIZE=6
PERT_INTERNAL_BATCH_SIZE=16

declare explainer_args=(
    "--explainer-name grad/saliency --eval-batch-size $EVAL_BATCH_SIZE"
    "--explainer-name grad/deeplift --eval-batch-size $EVAL_BATCH_SIZE"
    "--explainer-name grad/input_x_gradient --eval-batch-size $EVAL_BATCH_SIZE"
    "--explainer-name grad/guided_backprop --eval-batch-size $EVAL_BATCH_SIZE"
    "--explainer-name grad/gradient_shap --eval-batch-size 1 --internal_batch_size $GRAD_INTERNAL_BATCH_SIZE"
    "--explainer-name grad/integrated_gradients --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $GRAD_INTERNAL_BATCH_SIZE"
    "--explainer-name grad/deeplift_shap --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $GRAD_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/feature_ablation --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/lime --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/kernel_shap --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/occlusion --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
)

PROJECT_NAME="docxeval"
EXP_NAME="experiment_02_cls"
TASK="sequence_classification"
COMMON_ARGS="--total_samples 1000 --compute-metrics True --only-load-cached-explanations False"

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