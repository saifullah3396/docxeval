#!/bin/bash
set -e

PATH_TO_DATA_DIR="data"
EXPLANATIONS_DIR=experiment_02_qa
ATTN_EXPLANATIONS_DIR=experiment_03_qa

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

declare explainer_args=(
    "--explainer-name grad/saliency --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name grad/deeplift --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name grad/input_x_gradient --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name grad/guided_backprop --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name grad/gradient_shap --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name grad/integrated_gradients --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name grad/deeplift_shap --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name perturbation/feature_ablation --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name perturbation/lime --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name perturbation/kernel_shap --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name perturbation/occlusion --explanations-dir $EXPLANATIONS_DIR"
    "--explainer-name attn/raw_attention --explanations-dir $ATTN_EXPLANATIONS_DIR"
    "--explainer-name attn/attention_rollout --explanations-dir $ATTN_EXPLANATIONS_DIR"
)

PROJECT_NAME="docxeval"
EXP_NAME="explanation_analysis" 
TASK="question_answering"

declare -a CONFIGS=()
for eval_arg in "${eval_args[@]}"; do
    DATASET_NAME=$(sed -n 's/.*--dataset-name \([^ ]*\).*/\1/p' <<< "$eval_arg")
    DATASET_NAME=${DATASET_NAME//\//_}  # replace '/' with '_'
    MODEL_NAME=$(sed -n 's/.*--model-name \([^ ]*\).*/\1/p' <<< "$eval_arg") 

    for explainer_arg in "${explainer_args[@]}"; do
        EXPLAINER_NAME=$(sed -n 's/.*--explainer-name \([^ ]*\).*/\1/p' <<< "$explainer_arg")
        EXPLAINER_NAME=${EXPLAINER_NAME// /_}  # replace
        EXPLAINER_DIR=$(sed -n 's/.*--explanations-dir \([^ ]*\).*/\1/p' <<< "$explainer_arg")
        EXPLAINER_DIR=${EXPLAINER_DIR// /_}  # replace
        job_name="${EXP_NAME}_${DATASET_NAME}_${MODEL_NAME}_${EXPLAINER_NAME}"
        script="python src/docxeval/explanation_analyzer/${TASK}.py --project-name "$PROJECT_NAME" --exp-name "viz_main_v2" --output-dir ../outputs/ $eval_arg $explainer_arg --explanations-dir ${EXPLAINER_DIR} --mode batch --viz-batch-size 1 --n-batches 10"
        CONFIGS+=("$job_name $script")
    done 
done
