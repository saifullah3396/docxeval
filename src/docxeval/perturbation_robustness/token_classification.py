# noqa
import itertools
from pathlib import Path
from typing import Generator, Literal

import fire
from atria_datasets.api.datasets import load_dataset_config
from atria_insights.baseline_generators._sequence import SequenceBaselineGeneratorConfig
from atria_insights.configs.explanation_task_config import (
    LoggingConfig,
)
from atria_insights.feature_segmentors._image import GridSegmenterConfig
from atria_insights.feature_segmentors._sequence import (
    SequenceFeatureMaskSegmentorConfig,
)
from atria_insights.perturbation_robustness._evaluator import (
    PerturbationRobustnessEvaluator,
)
from atria_insights.perturbation_robustness._task_config import (
    PerturbationRobustnessEvaluatorTaskConfig,
)
from atria_logger import get_logger
from atria_ml.configs import (
    DataConfig,
    RuntimeEnvConfig,
)
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_builders._common import ModelBuilderType
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform
from atria_transforms.core._tfs._base import DataTransform
from atria_transforms.tfs._image_transforms import StandardImageTransform
from atria_transforms.tfs._perturbation._feature_perturbation_transform import (
    SequenceFeaturePertubationTransform,
)

logger = get_logger(__name__)


def perturbation_transform_generator(
    model_name: str | None = None,
    only_eval_baseline: bool = False,
    **baseline_kwargs,
) -> Generator[DataTransform, None, None]:
    baseline_type_options_per_modality = (
        SequenceBaselineGeneratorConfig.baseline_types_per_modality()
    )

    yield SequenceFeaturePertubationTransform(
        feature_segmentor=SequenceFeatureMaskSegmentorConfig(
            image_segmentor=GridSegmenterConfig(cell_size=16)
        ),
        baseline_generator=SequenceBaselineGeneratorConfig(**baseline_kwargs),
        percent_features_perturbed=0.0,
        ignored_feature_ids=["token_type_ids"],
    )

    if only_eval_baseline:
        return

    if model_name in ["bert-base-uncased", "roberta-base"]:
        baseline_type_options_per_modality.pop("layout_ids")
        baseline_type_options_per_modality.pop("image")
    elif model_name in ["lilt-roberta-base"]:
        baseline_type_options_per_modality.pop("image")

    baseline_type_options_per_modality.pop("token_type_ids")
    keys = list(baseline_type_options_per_modality.keys())
    values = list(baseline_type_options_per_modality.values())

    for combination in itertools.product(*values):
        for percent_features_perturbed in [0.1, 0.25, 0.4, 0.8]:
            yield SequenceFeaturePertubationTransform(
                feature_segmentor=SequenceFeatureMaskSegmentorConfig(
                    image_segmentor=GridSegmenterConfig(cell_size=16)
                ),
                baseline_generator=SequenceBaselineGeneratorConfig(
                    **dict(zip(keys, combination)), **baseline_kwargs
                ),
                percent_features_perturbed=percent_features_perturbed,
                ignored_feature_ids=["token_type_ids"],
            )


def main(
    checkpoint_path: str,
    data_dir: str | None = None,
    max_eval_samples: int = 1000,
    project_name: str = "docxeval",
    dataset_name: str = "funsd",
    model_name: str = "bert-base-uncased",
    tokenizer_name: str = "bert-base-uncased",
    builder_type: ModelBuilderType = ModelBuilderType.atria,
    exp_name: str = "robustness_eval_token_cls_00",
    output_dir: str = "./outputs",
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "standard",
    image_size: int = 224,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 8,
    access_token: str | None = None,
    use_segment_level_bboxes: bool = False,
    only_eval_baseline: bool = False,
):
    assert Path(
        checkpoint_path
    ).exists(), f"Checkpoint path {checkpoint_path} does not exist."
    image_transform = StandardImageTransform(
        stats=stats, resize_width=image_size, resize_height=image_size
    )
    mean, std = image_transform._get_stats()

    # for sweep_config in tqdm.tqdm(sweep_configs, desc="Running sweep configs"):
    config = PerturbationRobustnessEvaluatorTaskConfig(
        env=RuntimeEnvConfig(
            project_name=project_name,
            exp_name=exp_name,
            dataset_name=dataset_name.replace("/", "_"),
            model_name=model_name,
            output_dir=output_dir,
        ),
        logging=LoggingConfig(logging_steps=10, refresh_rate=10),
        data=DataConfig(
            data_dir=data_dir,
            access_token=access_token,
            dataset_config=load_dataset_config(dataset_name),
            num_workers=num_workers,
            num_processes=num_workers,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            split_ratio=0.95,
        ),
        model_pipeline=load_model_pipeline_config(
            "token_classification",
            model=ModelConfig(
                model_name_or_path=model_name,
                builder_type=builder_type,
                model_type="token_classification",
            ),
            train_transform=load_transform(
                "document_processor/token_classification",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                image_transform=image_transform,
                overflow_strategy="return_random",
                use_segment_level_bboxes=use_segment_level_bboxes,
            ),
            eval_transform=load_transform(
                "document_processor/token_classification",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                image_transform=image_transform,
                overflow_strategy="return_all",
                use_segment_level_bboxes=use_segment_level_bboxes,
            ),
        ),
        n_runs_per_perturbation=3,
        max_eval_samples=max_eval_samples,
    )
    evaluator = PerturbationRobustnessEvaluator(
        config=config,
        perturbation_transform_generator=perturbation_transform_generator(
            model_name=model_name,
            only_eval_baseline=only_eval_baseline,
            image_mean=mean,
            image_std=std,
        ),
        checkpoint_path=checkpoint_path,
    )
    evaluator.run()


if __name__ == "__main__":
    fire.Fire(main)
