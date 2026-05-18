#!/bin/bash

PATH_TO_DATA_DIR="data"

declare -a datasets=(
    "--dataset-name due_benchmark/DocVQA --data_dir $PATH_TO_DATA_DIR/due_benchmark"
)

declare -a model_args=(
    "--model-name bert-base-uncased --builder-type atria --tokenizer-name bert-base-uncased  --save_images_in_preprocess False --save_bboxes_in_preprocess False"
    "--model-name roberta-base --builder-type atria --tokenizer-name roberta-base --save_images_in_preprocess False --save_bboxes_in_preprocess False"
    "--model-name lilt-roberta-base --builder-type atria --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base --save_images_in_preprocess False --save_bboxes_in_preprocess True"
    "--model-name layoutlmv3-base --builder-type atria --tokenizer-name microsoft/layoutlmv3-base --use_segment_level_bboxes True --save_images_in_preprocess True --save_bboxes_in_preprocess True"
)

PROJECT_NAME="docxeval"
EXP_NAME="experiment_00_qa"
TASK="question_answering"

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
