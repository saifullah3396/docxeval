#!/bin/bash

PATH_TO_DATA_DIR="data"

declare -a datasets=(
    "--dataset-name cord --data_dir $PATH_TO_DATA_DIR/cord/"
    "--dataset-name funsd --data_dir $PATH_TO_DATA_DIR/funsd/"
    "--dataset-name sroie --data_dir $PATH_TO_DATA_DIR/sroie/"
    "--dataset-name wild_receipts --data_dir $PATH_TO_DATA_DIR/wild_receipts/"
)

declare -a model_args=(
    "--model-name bert-base-uncased --builder-type atria --tokenizer-name bert-base-uncased"
    "--model-name roberta-base --builder-type atria --tokenizer-name roberta-base"
    "--model-name lilt-roberta-base --builder-type atria --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base"
    "--model-name layoutlmv3-base --builder-type atria --tokenizer-name microsoft/layoutlmv3-base --use_segment_level_bboxes True"
)

PROJECT_NAME="docxeval"
EXP_NAME="experiment_00_ser"
TASK="token_classification"

declare -a CONFIGS=()
for dataset_args in "${datasets[@]}"; do
    for model_arg in "${model_args[@]}"; do
        dataset_name=$(echo "$dataset_args" | grep -oP '(?<=--dataset-name )[^ ]+' | tr '/' '_')
        model_name=$(echo "$model_arg" | grep -oP '(?<=--model-name )[^ ]+')
        job_name="${dataset_name}_${model_name}"
        script="python src/docxeval/trainers/${TASK}.py --project-name "$PROJECT_NAME" --exp-name "$EXP_NAME" --output-dir ../outputs/ $model_arg $dataset_args"
        CONFIGS+=("$job_name $script")
    done
done
