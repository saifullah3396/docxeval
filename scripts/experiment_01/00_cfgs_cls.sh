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


PROJECT_NAME="docxeval"
EXP_NAME="experiment_01_cls"
TASK="sequence_classification"

declare -a CONFIGS=()
for eval_arg in "${eval_args[@]}"; do
    DATASET_NAME=$(sed -n 's/.*--dataset-name \([^ ]*\).*/\1/p' <<< "$eval_arg")
    DATASET_NAME=${DATASET_NAME//\//_}  # replace '/' with '_'
    MODEL_NAME=$(sed -n 's/.*--model-name \([^ ]*\).*/\1/p' <<< "$eval_arg") 
    job_name="${EXP_NAME}_${DATASET_NAME}_${MODEL_NAME}"
    script="python src/docxeval/perturbation_robustness/${TASK}.py --project-name "$PROJECT_NAME" --exp-name "$EXP_NAME" --output-dir ../outputs/ $eval_arg "
    CONFIGS+=("$job_name $script")
done
