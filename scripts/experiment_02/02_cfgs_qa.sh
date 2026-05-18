#!/bin/bash
set -e


PATH_TO_DATA_DIR="data"

DOCVQA_CONFIG="--dataset-name due_benchmark/DocVQA --data_dir $PATH_TO_DATA_DIR/due_benchmark"

BERT_ARGS="--model-name bert-base-uncased --builder-type atria --tokenizer-name bert-base-uncased  --save_images_in_preprocess False --save_bboxes_in_preprocess False"
ROBERTA_ARGS="--model-name roberta-base --builder-type atria --tokenizer-name roberta-base --save_images_in_preprocess False --save_bboxes_in_preprocess False"
LILT_ROBERTA_ARGS="--model-name lilt-roberta-base --builder-type atria --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base --save_images_in_preprocess False --save_bboxes_in_preprocess True"
LAYOUTLMV3_ARGS="--model-name layoutlmv3-base --builder-type atria --tokenizer-name microsoft/layoutlmv3-base  --use_segment_level_bboxes True --save_images_in_preprocess True --save_bboxes_in_preprocess True"

declare -a eval_args=(
    "$DOCVQA_CONFIG $BERT_ARGS --checkpoint_path ../outputs/experiment_00_qa/due_benchmark_DocVQA/bert-base-uncased/checkpoints/<checkpoint_name>.pt"
    "$DOCVQA_CONFIG $ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_qa/due_benchmark_DocVQA/roberta-base/checkpoints/<checkpoint_name>.pt"
    "$DOCVQA_CONFIG $LILT_ROBERTA_ARGS --checkpoint_path ../outputs/experiment_00_qa/due_benchmark_DocVQA/lilt-roberta-base/checkpoints/<checkpoint_name>.pt"
    "$DOCVQA_CONFIG $LAYOUTLMV3_ARGS --checkpoint_path ../outputs/experiment_00_qa/due_benchmark_DocVQA/layoutlmv3-base/checkpoints/<checkpoint_name>.pt"
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
    "--explainer-name grad/integrated_gradients --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $GRAD_INTERNAL_BATCH_SIZE --grad_batch_size 16"
    "--explainer-name grad/deeplift_shap --eval-batch-size 1 --internal_batch_size 6 --grad_batch_size 16"
    "--explainer-name perturbation/feature_ablation --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/lime --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/kernel_shap --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
    "--explainer-name perturbation/occlusion --eval-batch-size $EVAL_BATCH_SIZE --internal_batch_size $PERT_INTERNAL_BATCH_SIZE"
)


PROJECT_NAME="docxeval"
EXP_NAME="experiment_02_qa"
TASK="question_answering"
COMMON_ARGS="--total_samples 1000 --compute-metrics True"

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