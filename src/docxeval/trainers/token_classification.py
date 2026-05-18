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
    dataset_name: str = "funsd",
    model_name: str = "bert-base-uncased",
    tokenizer_name: str = "bert-base-uncased",
    builder_type: ModelBuilderType = ModelBuilderType.atria,
    exp_name: str = "train_token_cls_00",
    output_dir: str = "./outputs",
    stats: Literal["imagenet", "standard", "openai_clip", "custom"] = "standard",
    image_size: int = 224,
    max_epochs: int = 100,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    num_workers: int = 8,
    seed: int = 42,
    optim: str = "adamw",
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    access_token: str | None = None,
    use_segment_level_bboxes: bool = False,
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
            "token_classification",
            model=ModelConfig(
                model_name_or_path=model_name,
                builder_type=builder_type,
                model_type="token_classification",
                # model_kwargs=dict(pretrained=False),
            ),
            train_transform=load_transform(
                "document_processor/token_classification",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_random",
                use_segment_level_bboxes=use_segment_level_bboxes,
            ),
            eval_transform=load_transform(
                "document_processor/token_classification",
                hf_processor={
                    "tokenizer_name": tokenizer_name,
                },
                image_transform=StandardImageTransform(
                    stats=stats, resize_width=image_size, resize_height=image_size
                ),
                overflow_strategy="return_all",
                use_segment_level_bboxes=use_segment_level_bboxes,
            ),
        ),
        # model_pipeline=load_model_pipeline_config(
        #     "token_classification",
        #     model=ModelConfig(
        #         model_name_or_path=model_name,
        #         builder_type=builder_type,
        #         model_type="token_classification",
        #     ),
        #     train_transform=load_transform(
        #         "token_classification_document_processor",
        #         hf_processor={
        #             "tokenizer_name": tokenizer_name,
        #         },
        #         image_transform=StandardImageTransform(
        #             stats=stats, resize_width=image_size, resize_height=image_size
        #         ),
        #         overflow_strategy="return_random",
        #         use_segment_level_bboxes=use_segment_level_bboxes,
        #     ),
        #     eval_transform=load_transform(
        #         "token_classification_document_processor",
        #         hf_processor={
        #             "tokenizer_name": tokenizer_name,
        #         },
        #         image_transform=StandardImageTransform(
        #             stats=stats, resize_width=image_size, resize_height=image_size
        #         ),
        #         overflow_strategy="return_all",
        #         use_segment_level_bboxes=use_segment_level_bboxes,
        #     ),
        # ),
        data=DataConfig(
            data_dir=data_dir,
            access_token=access_token,
            dataset_config=load_dataset_config(dataset_name),
            num_workers=num_workers,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            split_ratio=0.95,
            # preprocess_train_transform=load_transform(
            #     "document_tokenizer",
            #     hf_processor={
            #         "tokenizer_name": tokenizer_name,
            #     },
            #     use_segment_level_bboxes=use_segment_level_bboxes,
            #     image_size=(image_size, image_size),
            #     save_images=save_images_in_preprocess,
            #     save_bboxes=save_bboxes_in_preprocess,
            # ),
            # preprocess_eval_transform=load_transform(
            #     "document_tokenizer",
            #     hf_processor={
            #         "tokenizer_name": tokenizer_name,
            #     },
            #     use_segment_level_bboxes=use_segment_level_bboxes,
            #     image_size=(image_size, image_size),
            #     save_images=save_images_in_preprocess,
            #     save_bboxes=save_bboxes_in_preprocess,
            # ),
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
                monitored_metric="validation/seqeval/f1_score",
                patience=10,
                mode="max",
            ),
            model_checkpoint=ModelCheckpointConfig(
                monitored_metric="validation/seqeval/f1_score",
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
