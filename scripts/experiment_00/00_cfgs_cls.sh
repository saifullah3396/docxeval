#!/bin/bash

PATH_TO_DATA_DIR="data" 

declare -a datasets=(
    "--dataset-name tobacco3482/image_with_ocr --data-dir $PATH_TO_DATA_DIR/tobacco3482/"
    "--dataset-name rvlcdip/image_with_ocr --data-dir $PATH_TO_DATA_DIR/rvlcdip/ --train-batch-size 64 --eval-batch-size 64"
    "--dataset-name doclaynet/default --data-dir $PATH_TO_DATA_DIR/doclaynet"
)

declare -a model_args=(
    "--model-name bert-base-uncased --builder-type atria --tokenizer-name bert-base-uncased --save_images_in_preprocess False --save_bboxes_in_preprocess False"
    "--model-name roberta-base --builder-type atria --tokenizer-name roberta-base  --save_images_in_preprocess False --save_bboxes_in_preprocess False"
    "--model-name lilt-roberta-base --builder-type atria --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base --save_images_in_preprocess False --save_bboxes_in_preprocess True"
    "--model-name layoutlmv3-base --builder-type atria --tokenizer-name microsoft/layoutlmv3-base  --save_images_in_preprocess True --save_bboxes_in_preprocess True"
)

PROJECT_NAME="docxeval"
EXP_NAME="experiment_00_cls"
TASK="sequence_classification"
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
