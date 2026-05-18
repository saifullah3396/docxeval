# noqa
from typing import Literal

import fire
from atria_datasets.api.datasets import load_dataset_config
from atria_insights.baseline_generators._feature_based import (
    FeatureBasedBaselineGeneratorConfig,
)
from atria_insights.baseline_generators._sequence import SequenceBaselineGeneratorConfig
from atria_insights.configs.explanation_task_config import (
    ExplanationTaskConfig,
    LoggingConfig,
)
from atria_insights.explainability_metrics import (  # noqa
    AOPCConfig,  # noqa
    CompletenessConfig,  # noqa
    ComplexityEntropyConfig,  # noqa
    ComplexitySConfig,  # noqa
    EffectiveComplexityConfig,  # noqa
    FaithfulnessCorrelationConfig,  # noqa
    FaithfulnessEstimateConfig,  # noqa
    InfidelityConfig,  # noqa
    MonotonicityConfig,  # noqa
    MonotonicityCorrAndNonSensConfig,  # noqa
    SensitivityMaxAvgConfig,  # noqa
    SensitivityNConfig,  # noqa
    SparsenessConfig,  # noqa
)
from atria_insights.explainers import (
    DeepLiftExplainerConfig,
    DeepLiftShapExplainerConfig,
    FeatureAblationExplainerConfig,
    GradientShapExplainerConfig,
    GuidedBackpropExplainerConfig,
    InputXGradientExplainerConfig,
    IntegratedGradientsExplainerConfig,
    KernelShapExplainerConfig,
    LimeExplainerConfig,
    OcclusionExplainerConfig,
    SaliencyExplainerConfig,
)
from atria_insights.feature_segmentors._image import GridSegmenterConfig
from atria_insights.feature_segmentors._sequence import (
    SequenceFeatureMaskSegmentorConfig,
)
from atria_insights.model_explainer import ModelExplainer
from atria_insights.model_pipelines._api import load_x_model_pipeline_config
from atria_insights.model_pipelines._common import ExplanationTargetStrategy
from atria_logger import get_logger
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
)
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_builders._common import ModelBuilderType
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform
from atria_transforms.tfs._image_transforms import StandardImageTransform

logger = get_logger(__name__)

_EXPLAINERS = {
    "grad/saliency": SaliencyExplainerConfig(),
    "grad/integrated_gradients": IntegratedGradientsExplainerConfig(n_steps=200),
    "grad/deeplift": DeepLiftExplainerConfig(),
    "grad/deeplift_shap": DeepLiftShapExplainerConfig(),
    "grad/gradient_shap": GradientShapExplainerConfig(n_samples=200),
    "grad/guided_backprop": GuidedBackpropExplainerConfig(),
    "grad/input_x_gradient": InputXGradientExplainerConfig(),
    "perturbation/feature_ablation": FeatureAblationExplainerConfig(
        weight_attributions=True
    ),
    "perturbation/kernel_shap": KernelShapExplainerConfig(
        n_samples=200, weight_attributions=True
    ),
    "perturbation/lime": LimeExplainerConfig(
        n_samples=200,
        weight_attributions=True,
    ),
    "perturbation/occlusion": OcclusionExplainerConfig(),
}

_DEFAULT_DEEPSHAP_BASELINES_GENERATOR_CONFIG = FeatureBasedBaselineGeneratorConfig(
    num_baselines=100
)

_DEFAULT_FEATURE_SEGMENTOR_CONFIG = SequenceFeatureMaskSegmentorConfig(
    image_segmentor=GridSegmenterConfig(cell_size=16)
)

_METRICS = {
    "axiomatic/completeness": CompletenessConfig(),
    "axiomatic/monotonicity_corr_and_non_sens": MonotonicityCorrAndNonSensConfig(
        n_perturbations_per_feature=1,
        max_features_processed_per_batch=100,
        percentage_feature_removal_per_step=0.01,  # 1% of the features will be removed together in each step
        zero_attribution_threshold=1.0e-3,
        zero_variance_threshold=1.0e-1,
        use_percentage_attribution_threshold=True,
        return_ratio=True,
        show_progress=True,
    ),
    "complexity/complexity_entropy": ComplexityEntropyConfig(group_features=True),
    "complexity/complexity_s": ComplexitySConfig(group_features=True, eps=1.0e-03),
    "complexity/effective_complexity": EffectiveComplexityConfig(
        n_perturbations_per_feature=1,
        max_features_processed_per_batch=100,
        percentage_feature_removal_per_step=0.01,
        zero_variance_threshold=1.0e-1,
        return_ratio=True,
        show_progress=True,
    ),
    "complexity/sparseness": SparsenessConfig(group_features=True),
    "faithfulness/aopc": AOPCConfig(
        total_feature_bins=200,
        n_random_perms=3,
        max_features_processed_per_batch=100,
        show_progress=True,
    ),
    **{
        f"faithfulness/faithfulness_correlation_{perc}": FaithfulnessCorrelationConfig(
            n_perturb_samples=200,
            max_examples_per_batch=100,
            percent_features_perturbed=perc / 100,
            show_progress=True,
        )
        for perc in [20, 40, 60]
    },
    "faithfulness/faithfulness_estimate": FaithfulnessEstimateConfig(
        max_features_processed_per_batch=100,
        percentage_feature_removal_per_step=0.01,
        show_progress=True,
    ),
    "faithfulness/infidelity": InfidelityConfig(
        max_examples_per_batch=100,
        n_perturb_samples=200,
        perturbation_noise_scale=0.1,
    ),
    "faithfulness/monotonicity": MonotonicityConfig(
        max_features_processed_per_batch=100,
        percentage_feature_removal_per_step=0.01,
    ),
    **{
        f"faithfulness/sensitivity_n_{perc}": SensitivityNConfig(
            n_features_perturbed=perc / 100,
            max_examples_per_batch=100,
            n_perturb_samples=200,
            normalize=True,
        )
        for perc in [20, 40, 60]
    },
    "robustness/sensitivity_max_and_avg": SensitivityMaxAvgConfig(
        perturb_radius=0.02,
        n_perturb_samples=10,
        norm_ord="fro",
        max_examples_per_batch=1,
    ),
}

# these are latest extracted by running analysis/task_wise_modality_rankings.py script
_BASELINE_TYPES = {
    "bert-base-uncased": {
        "token_ids": "mask_token_id",
        "token_type_ids": "zero",
        "position_ids": "zero",
    },
    "roberta-base": {
        "token_ids": "pad_token_id",
        "token_type_ids": "zero",
        "position_ids": "zero",
    },
    "lilt-roberta-base": {
        "token_ids": "pad_token_id",
        "token_type_ids": "zero",
        "position_ids": "zero",
        "layout_ids": "pad_token_id",
    },
    "layoutlmv3-base": {
        "token_ids": "pad_token_id",
        "token_type_ids": "zero",
        "position_ids": "zero",
        "layout_ids": "zero",
        "image": "random",
    },
}


def main(
    data_dir: str | None = None,
    checkpoint_path: str | None = None,
    project_name: str = "docxeval",
    dataset_name: str = "due_benchmark/DocVQA",
    model_name: str = "bert-base-uncased",
    tokenizer_name: str = "bert-base-uncased",
    explainer_name: str = "grad/saliency",
    builder_type: ModelBuilderType = ModelBuilderType.atria,
    exp_name: str = "explain_qa_00",
    output_dir: str = "./outputs",
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "standard",
    image_size: int = 224,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    internal_batch_size: int = 4,
    grad_batch_size: int = 4,
    num_workers: int = 8,
    seed: int = 42,
    total_samples: int = 100,
    compute_metrics: bool = False,
    access_token: str | None = None,
    use_segment_level_bboxes: bool = False,
    save_images_in_preprocess: bool = False,
    save_bboxes_in_preprocess: bool = False,
    compute_features_only: bool = False,
    only_load_cached_explanations: bool = False,
):
    assert explainer_name in _EXPLAINERS, f"Explainer {explainer_name} not recognized."

    image_transform = StandardImageTransform(
        stats=stats, resize_width=image_size, resize_height=image_size
    )
    mean, std = image_transform._get_stats()
    explainer_baseline_generator = SequenceBaselineGeneratorConfig(
        **_BASELINE_TYPES[model_name],  # type: ignore
        image_mean=mean,
        image_std=std,
    )
    metric_baseline_generator = SequenceBaselineGeneratorConfig(
        **_BASELINE_TYPES[model_name],  # type: ignore
        image_mean=mean,
        image_std=std,
    )

    # for deeeplift shap, we use feature based baseline generator with multiple baselines
    if explainer_name in ["grad/deeplift_shap"]:
        explainer_baseline_generator = _DEFAULT_DEEPSHAP_BASELINES_GENERATOR_CONFIG

    config = ExplanationTaskConfig(
        env=RuntimeEnvConfig(
            project_name=project_name,
            exp_name=exp_name,
            dataset_name=dataset_name.replace("/", "_"),
            model_name=model_name,
            output_dir=output_dir,
            seed=seed,
        ),
        logging=LoggingConfig(logging_steps=1, refresh_rate=1),
        data=DataConfig(
            data_dir=data_dir,
            access_token=access_token,
            dataset_config=load_dataset_config(dataset_name),
            num_workers=num_workers,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            split_ratio=0.95,
            preprocess_train_transform=load_transform(
                "document_tokenizer/question_answering",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                use_segment_level_bboxes=use_segment_level_bboxes,
                resize_image=(image_size, image_size),
                load_image=save_images_in_preprocess,
                load_bboxes=save_bboxes_in_preprocess,
                ignore_no_answer_qa_pair=True,
            ),
            preprocess_eval_transform=load_transform(
                "document_tokenizer/question_answering",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                use_segment_level_bboxes=use_segment_level_bboxes,
                resize_image=(image_size, image_size),
                load_image=save_images_in_preprocess,
                load_bboxes=save_bboxes_in_preprocess,
                ignore_no_answer_qa_pair=False,
            ),
        ),
        x_model_pipeline=load_x_model_pipeline_config(
            "question_answering",
            model_pipeline=load_model_pipeline_config(
                "question_answering",
                model=ModelConfig(
                    model_name_or_path=model_name,
                    builder_type=builder_type,
                    model_type="question_answering",
                ),
                train_transform=load_transform(
                    "document_processor/question_answering",
                    hf_processor={"tokenizer_name": tokenizer_name},
                    image_transform=StandardImageTransform(
                        stats=stats, resize_width=image_size, resize_height=image_size
                    ),
                    overflow_strategy="return_random",
                    use_segment_level_bboxes=use_segment_level_bboxes,
                    ignore_no_answer_qa_pair=True,
                ),
                eval_transform=load_transform(
                    "document_processor/question_answering",
                    hf_processor={"tokenizer_name": tokenizer_name},
                    image_transform=StandardImageTransform(
                        stats=stats, resize_width=image_size, resize_height=image_size
                    ),
                    overflow_strategy="return_all",
                    ignore_no_answer_qa_pair=False,
                ),
            ),
            feature_segmentor=_DEFAULT_FEATURE_SEGMENTOR_CONFIG,
            baseline_generator=explainer_baseline_generator,
            metric_baseline_generator=metric_baseline_generator,
            explainer=_EXPLAINERS[explainer_name],
            explainability_metrics=_METRICS,
            explanation_target_strategy=ExplanationTargetStrategy.predicted,
            iterative_computation=False,
            internal_batch_size=internal_batch_size,
            grad_batch_size=grad_batch_size,
            throw_on_load_mismatch=only_load_cached_explanations,
        ),
        enable_outputs_caching=True,
    )
    model_explainer = ModelExplainer(config=config, checkpoint_path=checkpoint_path)
    model_explainer.run(
        total_samples=total_samples,
        compute_metrics=compute_metrics,
        compute_features_only=compute_features_only,
    )


if __name__ == "__main__":
    fire.Fire(main)
