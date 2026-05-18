from typing import Literal

import fire
from atria_datasets.api.datasets import load_dataset_config
from atria_ml.configs import (
    DataConfig,
    EarlyStoppingConfig,
    ModelCheckpointConfig,
    RuntimeEnvConfig,
    TrainerConfig,
    TrainingTaskConfig,
    WarmupConfig,
)
from atria_ml.optimizers._api import load_optimizer_config
from atria_ml.task_pipelines._trainer import Trainer
from atria_models.api.models import load_model_pipeline_config
from atria_models.core.model_builders._common import ModelBuilderType
from atria_models.core.model_pipelines._common import ModelConfig
from atria_transforms.api.tfs import load_transform
from atria_transforms.tfs._image_transforms import StandardImageTransform


def main(
    data_dir: str | None = None,
    project_name: str = "docxeval",
    dataset_name: str = "due_benchmark/DocVQA",
    model_name: str = "bert-base-uncased",
    tokenizer_name: str = "bert-base-uncased",
    builder_type: ModelBuilderType = ModelBuilderType.atria,
    exp_name: str = "train_qa_cls_00",
    output_dir: str = "./outputs",
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "standard",
    image_size: int = 224,
    max_epochs: int = 50,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 8,
    seed: int = 42,
    optim: str = "adamw",
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    access_token: str | None = None,
    use_segment_level_bboxes: bool = False,
    save_images_in_preprocess: bool = False,
    save_bboxes_in_preprocess: bool = False,
):
    config = TrainingTaskConfig(
        env=RuntimeEnvConfig(
            project_name=project_name,
            exp_name=exp_name,
            dataset_name=dataset_name.replace("/", "_"),
            model_name=model_name,
            output_dir=output_dir,
            seed=seed,
        ),
        model_pipeline=load_model_pipeline_config(
            "question_answering",
            model=ModelConfig(
                model_name_or_path=model_name,
                builder_type=builder_type,
                model_type="question_answering",
            ),
            train_transform=load_transform(
                "document_processor/question_answering",
                hf_processor={
                    "tokenizer_name": "bert-base-uncased",
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_random",
                use_segment_level_bboxes=use_segment_level_bboxes,
                ignore_no_answer_qa_pair=True,
            ),
            eval_transform=load_transform(
                "document_processor/question_answering",
                hf_processor={
                    "tokenizer_name": "bert-base-uncased",
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_all",
                ignore_no_answer_qa_pair=False,
            ),
        ),
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
        trainer=TrainerConfig(
            max_epochs=max_epochs,
            optimizer=load_optimizer_config(
                optimizer_name=optim,
                lr=lr,
                weight_decay=weight_decay,
            ),
            warmup=WarmupConfig(
                warmup_steps=warmup_steps,
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                monitored_metric="validation/due_eval/ANLS",
                patience=10,
                mode="max",
            ),
            model_checkpoint=ModelCheckpointConfig(
                monitored_metric="validation/due_eval/ANLS",
                mode="max",
            ),
        ),
        do_train=True,
        do_validation=True,
        do_test=True,
    )
    trainer = Trainer(config=config)
    trainer.run()


if __name__ == "__main__":
    fire.Fire(main)
