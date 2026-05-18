import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.registry.image_classification.cifar10 import Cifar10  # noqa
from atria_insights.configs.explanation_task_config import ExplanationTaskConfig
from atria_insights.explainability_metrics._torchxai._base import ExplainabilityMetric
from atria_insights.model_pipelines._model_pipeline import ExplainableModelPipeline
from atria_insights.storage.sample_cache_managers._explanation_state import (
    ExplanationStateCacher,
)
from atria_insights.storage.sample_cache_managers._metric_data_cacher import (
    MetricDataCacher,
)
from atria_logger._api import enable_file_logging, get_logger
from atria_ml.data_pipeline._data_pipeline import DataPipeline
from atria_ml.task_pipelines._utilities import _get_env_info, _initialize_torch
from atria_ml.training.engines.utilities import _format_metrics_for_logging
from atria_models.core.types.model_outputs import QAModelOutput
from atria_types._common import DatasetSplitType
from ignite.handlers import TensorboardLogger
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from docxeval.explanation_analyzer.explanation_summarizer.explanation_summarizer import (
    ExplanationSummarizer,
)
from docxeval.explanation_analyzer.explanation_summarizer.explanation_summary import (
    ExplanationSummary,
)
from docxeval.explanation_analyzer.utils.viz import plot_modality_diagnostics
from docxeval.explanation_analyzer.utils.x_metric_post_processors import (
    METRIC_POST_PROCESSORS,
    MetricPostProcessor,
)


class SummaryBuilderDataset(Dataset):
    def __init__(
        self,
        analyzer: "ModelExplanationAnalyzer",
        keys: list[str],
        sample_summary_root: Path,
        explanation_summary_root: Path,
        load_explanations: bool = True,
    ):
        self.analyzer = analyzer
        self.keys = keys
        self.sample_summary_root = sample_summary_root
        self.explanation_summary_root = explanation_summary_root
        self.load_explanations = load_explanations

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[str, bool, str | None]:
        sample_id = self.keys[idx]
        # sample_summary_exists = SampleSummary.exists(
        #     sample_id, self.sample_summary_root
        # )
        explanation_summary_exists = ExplanationSummary.exists(
            sample_id, self.explanation_summary_root
        )
        # if sample_summary_exists and explanation_summary_exists:
        #     return sample_id, True, None
        if explanation_summary_exists:
            return sample_id, True, None

        try:
            summary = self.analyzer.get_explanation_summarizer(
                sample_id=sample_id,
                load_explanations=self.load_explanations,
            )
        except ValueError as e:
            logger.error(
                f"Error getting explanation summarizer for sample {sample_id}: {e}"
            )
            return sample_id, False, f"Error getting explanation summarizer: {e}"

        # sample_summary = summary.generate_sample_summary()
        explanation_summary = summary.generate_explanation_summary()

        # sample_summary.save_to_disk(self.sample_summary_root)
        explanation_summary.save_to_disk(self.explanation_summary_root)
        return sample_id, True, None


logger = get_logger(__name__)


@dataclass
class ModelExplanationAnalyzerState:
    data_pipeline: DataPipeline
    x_model_pipeline: ExplainableModelPipeline
    x_metrics: dict[str, ExplainabilityMetric]

    @property
    def dataset(self) -> Dataset:
        return self.data_pipeline.dataset


class ModelExplanationAnalyzerConfig(ExplanationTaskConfig):
    explanations_dir: str = "experiment_02_seq_cls"
    mode: Literal[
        "single",
        "batch",
        "diagnose",
        "save",
        "summarize_explanations",
        "prepare_metrics",
        "metric_corrs",
    ] = "prepare_metrics"
    n_batches: int = 1
    batch_size: int = 4
    force_recompute: bool = False

    @property
    def explanations_run_dir(self) -> Path:
        base_path = Path(self.env.output_dir) / self.explanations_dir
        if self.env.dataset_name is not None:
            base_path = base_path / self.env.dataset_name
        if self.env.model_name is not None:
            base_path = base_path / self.env.model_name
        return base_path


class ModelExplanationAnalyzer:
    def __init__(
        self,
        config: ModelExplanationAnalyzerConfig,
        local_rank: int = 0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self._config = config
        self._checkpoint_path: str | Path | None = None
        if checkpoint_path is not None:
            self._checkpoint_path = checkpoint_path
            assert Path(
                self._checkpoint_path
            ).exists(), f"Checkpoint path {checkpoint_path} does not exist."
        self._run_dir = Path(self._config.env.run_dir)

        self._explanations_run_dir = Path(self._config.explanations_run_dir)
        subdir = os.listdir(self._explanations_run_dir)
        if len(subdir) > 0:
            assert (
                len(subdir) == 1
            ), f"Expected only one subdirectory in run_dir {self._explanations_run_dir}, but found {subdir}"
            self._explanations_run_dir = self._explanations_run_dir / subdir[0]
            self._run_dir = self._run_dir / subdir[0]

        self._run_dir = (
            self._run_dir / self._config.x_model_pipeline.explainer.type.split("/")[-1]
        )
        Path(self._run_dir).mkdir(parents=True, exist_ok=True)
        self._model_output_cache_dir = Path(self._run_dir) / "model_outputs_cache"
        self._model_output_cache_dir.mkdir(parents=True, exist_ok=True)
        self._state: ModelExplanationAnalyzerState | None = None

    def _model_output_cache_path(self, sample_id: str) -> Path:
        # Use a stable hash to avoid filesystem issues with sample_id characters.
        sample_hash = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        return self._model_output_cache_dir / f"{sample_hash}.pt"

    @property
    def state(self) -> ModelExplanationAnalyzerState:
        if self._state is None:
            self._state = self._build()
        return self._state

    def _initialize_runtime(self, local_rank: int) -> None:
        # Log system information
        env_info = _get_env_info()

        # initialize training
        _initialize_torch(
            seed=self._config.env.seed, deterministic=self._config.env.deterministic
        )

        # initialize torch device (cpu or gpu)
        self._device = local_rank

        # log env info and run configuration
        logger.info(
            f"Environment info:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(env_info)), indent=4)}"
        )
        logger.info(
            f"Run configuration:\n{yaml.dump(OmegaConf.to_container(OmegaConf.create(self._config.to_dict())), indent=4)}"
        )
        logger.info(f"Seed set to {self._config.env.seed} on device: {self._device}")

    def _setup_logging(self) -> TensorboardLogger | None:
        import ignite.distributed as idist

        if idist.get_rank() == 0:
            enable_file_logging(str(Path(self._run_dir) / "run.log"))
        return None

    def _build(self, local_rank: int = 0) -> ModelExplanationAnalyzerState:
        self._initialize_runtime(local_rank=local_rank)

        # setup logging
        self._setup_logging()

        # build dataset
        dataset = self._config.data.build_dataset()

        # preprocess dataset splits
        if (
            self._config.data.preprocess_train_transform is not None
            and self._config.data.preprocess_eval_transform is not None
        ):
            # process the dataset with a custom transform
            dataset = dataset.process_dataset(
                train_transform=self._config.data.preprocess_train_transform,
                eval_transform=self._config.data.preprocess_eval_transform,
                max_cache_image_size=self._config.data.preprocess_max_cache_image_size,
                num_processes=self._config.data.num_processes,
                processed_data_dir=self._config.data.data_dir,
            )

        # load labels
        labels = dataset.metadata.dataset_labels

        # log dataset info
        logger.info(f"Dataset:\n{dataset}")

        # see if feature baseline generator is attached, then we updates its path
        if self._config.x_model_pipeline.baseline_generator.type == "feature_based":
            # hard coded for now to the path where the features will be stored
            self._config.x_model_pipeline.baseline_generator.unsafe_update(
                features_path=str(Path(self._explanations_run_dir) / "features.hdf5")
            )

        # build model pipelines
        # force model wkargs to have pretrained = False to save load time
        x_model_pipeline = self._config.x_model_pipeline.build(
            labels=labels, persist_to_disk=False, cache_dir=self._explanations_run_dir
        )

        # get the explanation state cacher
        self._explainer_dir = (
            Path(self._explanations_run_dir)
            / self._config.x_model_pipeline.explainer.type.split("/")[-1]
        )
        self._explanations_cacher = ExplanationStateCacher(
            cache_dir=self._explainer_dir,
            config=self._config.x_model_pipeline,
            load_existing=True,
        )

        assert self._explanations_cacher is not None, (
            "Explanations cacher is not initialized in the model pipeline. Explanation Analyzer only works on "
            "precomputed cached explanations."
        )
        # log model pipeline
        logger.info(x_model_pipeline.summarize())

        # load model pipeline
        if self._checkpoint_path is not None:
            logger.info(
                f"Loading model pipeline from checkpoint: {self._checkpoint_path}"
            )
            x_model_pipeline._model_pipeline.load_checkpoint(self._checkpoint_path)

        # get model transforms
        train_transform = x_model_pipeline.config.model_pipeline.train_transform
        eval_transform = x_model_pipeline.config.model_pipeline.eval_transform
        try:
            dataset.train.output_transform = train_transform
            dataset.validation.output_transform = eval_transform
            dataset.test.output_transform = eval_transform
        except SplitNotFoundError:
            logger.warning(
                "One or more dataset splits not found while setting output transforms."
            )

        # build data pipeline
        data_pipeline = DataPipeline(dataset=dataset)

        # set model pipeline to device and eval
        x_model_pipeline._model_pipeline.ops.to_device(self._device)
        x_model_pipeline._model_pipeline.ops.eval()

        return ModelExplanationAnalyzerState(
            data_pipeline=data_pipeline,
            x_model_pipeline=x_model_pipeline,
            x_metrics=x_model_pipeline.build_metrics(device=self._device),
        )

    @torch.no_grad()
    def predict_step(self, sample_id: str):
        # dataset
        dataset_iterator = (
            self.state.data_pipeline.dataset.test
            if self.state.data_pipeline.dataset.split_exists(DatasetSplitType.test)
            else self.state.data_pipeline.dataset.validation
        )

        # fetch the data sample
        # we need original sample without transformation first
        output_transform = dataset_iterator.output_transform
        dataset_iterator.output_transform = None
        sample = dataset_iterator.fetch_sample_by_id(sample_id.split("_qa_")[0])

        # transformed sample
        transformed_sample = output_transform(sample)

        # we must extract the target transformed sample based on its id
        # this only used for qa otherwise has no effect
        transformed_sample = [
            x for x in transformed_sample if x.metadata.sample_id == sample_id
        ]

        # restore output transform
        dataset_iterator.output_transform = output_transform

        # Try loading cached model output first to avoid repeated forward passes.
        cache_path = self._model_output_cache_path(sample_id)
        if cache_path.exists():
            try:
                cache_payload = torch.load(
                    cache_path, map_location="cpu", weights_only=False
                )
                if cache_payload.get("sample_id") == sample_id and cache_payload.get(
                    "checkpoint_path"
                ) == str(self._checkpoint_path):
                    model_output = cache_payload["model_output"]

                    # update the sample id from transformed id
                    # this only used for qa otherwise has no effect
                    sample = sample.update(sample_id=sample_id)
                    return sample, transformed_sample, model_output
            except Exception as e:
                logger.warning(
                    "Could not load cached model output for sample %s: %s",
                    sample_id,
                    e,
                )

        # perform a forward pass on the sample to get the model outputs
        batch = (
            transformed_sample[0]
            .batch(transformed_sample)
            .ops.to_tensors()
            .ops.to(self._device)
        )
        model_output = self.state.x_model_pipeline._model_pipeline.predict_step(
            batch=batch
        )

        try:
            torch.save(
                {
                    "sample_id": sample_id,
                    "checkpoint_path": str(self._checkpoint_path),
                    "model_output": model_output,
                },
                cache_path,
            )
        except Exception as e:
            logger.warning(
                "Could not cache model output for sample %s: %s", sample_id, e
            )

        # update the sample id from transformed id
        # this only used for qa otherwise has no effect
        sample = sample.update(sample_id=sample_id)

        return sample, transformed_sample, model_output

    def get_explanation_summarizer(
        self, sample_id: str, load_explanations: bool = True
    ) -> ExplanationSummarizer:
        # sample step
        sample, transformed_sample, model_output = self.predict_step(sample_id)

        # get explanations for the sample
        explanation_state = self._explanations_cacher.load_sample(
            sample_key=sample_id,
            load_tensors=load_explanations,
        )

        # load explanation_metrics for the sample
        explanation_metrics = {}
        for metric_name, metric in self.state.x_metrics.items():
            cacher = MetricDataCacher(
                cache_dir=self._explainer_dir,
                config=metric.config,
                file_name=metric_name,
            )
            explanation_metrics[metric_name] = cacher.load_sample(
                sample_key=sample_id,
            )

        qa_targets = None
        if isinstance(model_output, QAModelOutput):
            # this is token levle target
            qa_targets = [
                model_output.start_logits.argmax().item(),
                model_output.end_logits.argmax().item(),
            ]

            word_ids = transformed_sample[0].word_ids

            if qa_targets[0] >= len(word_ids) or qa_targets[1] >= len(word_ids):
                logger.warning(
                    f"QA target indices {qa_targets} are out of bounds for word ids of length {len(word_ids)}. Setting targets to -100."
                )
                qa_targets = [-100, -100]
            else:
                # this is word level but words can be starting from a large value after tokenization
                qa_targets = [
                    word_ids[qa_targets[0]],
                    word_ids[qa_targets[1]],
                ]

            if qa_targets[-1] != -100 and qa_targets[0] != -100:
                # so we must subtract the minimum word id from the targets to get the actual target word indices
                min_word_id = 1e6
                for word_id, sequence_id in zip(
                    word_ids, transformed_sample[0].sequence_ids
                ):
                    if sequence_id == 1:  # only consider context words for qa
                        min_word_id = min(min_word_id, word_id)

                qa_targets = [target - min_word_id for target in qa_targets]
            else:
                qa_targets = [-100, -100]
        # prepare explanation summarizer
        explanation_summarizer = ExplanationSummarizer(
            sample=sample,
            transformed_sample=transformed_sample[0],
            model_output=model_output,
            explanation_state=explanation_state,
            explanation_metrics=explanation_metrics,
            dataset_name=self._config.env.dataset_name,
            model_name=self._config.env.model_name,
            explainer=self._config.x_model_pipeline.explainer.type,
            dataset_labels=self.state.dataset.metadata.dataset_labels,
            qa_targets=qa_targets,
        )
        return explanation_summarizer

    def load_and_cache_explanation_metadata(self, force_recompute: bool = False):
        cached_dataset_metrics_path = self._run_dir / "explanation_metadata.pkl"
        if cached_dataset_metrics_path.exists() and not force_recompute:
            logger.info(
                f"Loading cached explanation metrics summary from {cached_dataset_metrics_path}"
            )
            loaded_data = pd.read_pickle(cached_dataset_metrics_path)
            if len(loaded_data) > 0:
                return loaded_data

        logger.info("Preparing explanation metrics metadata for the entire dataset...")

        self._state = self._build()
        dataset_explanation_metadata = []
        sample_keys = self._explanations_cacher.list_sample_keys()
        for sample_id in tqdm.tqdm(
            sample_keys[:4],
            desc="Processing samples",
        ):
            try:
                summary = self.get_explanation_summarizer(
                    sample_id=sample_id, load_explanations=False
                )

                logger.debug("Processing sample ID: {}".format(sample_id))
                logger.debug(
                    "Sample metadata: {}".format(summary.get_explanation_metadata())
                )

                # get metadata
                dataset_explanation_metadata.append(summary.get_explanation_metadata())
            except Exception as e:
                logger.error(f"Error processing sample ID {sample_id}: {e}")
                continue

        # prepare explanation summary
        dataset_explanation_metadata_df = pd.DataFrame.from_records(
            dataset_explanation_metadata
        )

        # store to disk
        dataset_explanation_metadata_df.to_pickle(cached_dataset_metrics_path)

        return dataset_explanation_metadata_df

    def summarize_metrics(
        self, force_recompute: bool = False
    ) -> dict[str, dict[str, float] | str | int | float]:
        cached_metrics_summary_path = (
            self._run_dir / "summarized_explanation_metrics.json"
        )

        # get the explanation state cacher
        dataset_explanation_metadata_df = self.load_and_cache_explanation_metadata(
            force_recompute=force_recompute
        )

        # log dataset explanation metrics df
        logger.info(
            f"Dataset explanation metadata and metrics Dataframe:\n{dataset_explanation_metadata_df}"
        )

        # make sample id from index
        dataset_explanation_metadata_df["sample_id"] = (
            dataset_explanation_metadata_df.index
        )

        dataset_explanation_metadata_df["total_features"] = (
            dataset_explanation_metadata_df["monotonicity_corr_and_non_sens.n_features"]
        )

        # print all values of first sample
        logger.info("First sample explanation metadata and metrics:")
        for col in dataset_explanation_metadata_df.columns:
            logger.info(f"{col}: {dataset_explanation_metadata_df[col].shape}")

        summarized_metrics = {}
        for metric_processor in METRIC_POST_PROCESSORS:
            is_multi_target = isinstance(
                dataset_explanation_metadata_df["target"].iloc[0], list
            )
            summarized = metric_processor.summarize(
                dataset_explanation_metadata_df, is_multi_target=is_multi_target
            )
            summarized_metrics[metric_processor.metric_name] = summarized
            logger.info(
                f"Summarized metric {metric_processor.metric_name}: {summarized}"
            )

        # store to disk
        with open(cached_metrics_summary_path, "w") as f:
            formatted_metrics = _format_metrics_for_logging(summarized_metrics)
            json.dump(
                {
                    "dataset_name": self._config.env.dataset_name,
                    "model_name": self._config.env.model_name,
                    "explainer": self._config.x_model_pipeline.explainer.type,
                    "n_samples": len(dataset_explanation_metadata_df),
                    "metrics": {
                        **formatted_metrics,
                    },
                },
                f,
                indent=4,
            )

        with open(cached_metrics_summary_path, "r") as f:
            metrics_summary = json.load(f)
            return metrics_summary

    def _run_and_cache_predictions(self, sample_keys: list[str]):
        for sample_id in tqdm.tqdm(sample_keys, desc="Predict step (main process)"):
            try:
                # Try loading cached model output first to avoid repeated forward passes.
                cache_path = self._model_output_cache_path(sample_id)
                if cache_path.exists():
                    continue

                self.predict_step(sample_id)
            except Exception as e:
                logger.warning(
                    "Error during predict step for sample '%s': %s", sample_id, e
                )

    def compute_and_store_dataset_metrics(
        self,
        df: pd.DataFrame,
        run_dir: Path,
        dataset_name: str,
        model_name: str,
        explainer_type: str,
        metric_post_processors: list[MetricPostProcessor],
    ) -> None:
        logger.info(f"Dataset explanation metadata and metrics Dataframe:\n{df}")
        df["sample_id"] = df.index
        df["total_features"] = df["monotonicity_corr_and_non_sens.n_features"]

        logger.info("First sample explanation metadata and metrics:")
        for col in df.columns:
            logger.info(f"{col}: {df[col].shape}")

        is_multi_target = isinstance(df["targets"].iloc[0], list)

        # dataset-level aggregates
        summarized_metrics = {}
        for metric_processor in metric_post_processors:
            try:
                summarized = metric_processor.summarize(
                    df, is_multi_target=is_multi_target
                )
                summarized_metrics[metric_processor.metric_name] = summarized
                logger.info(
                    f"Summarized metric {metric_processor.metric_name}: {summarized}"
                )
            except Exception as e:
                logger.error(
                    f"Error summarizing metric {metric_processor.metric_name}: {e}"
                )

        # label-level aggregates (always, single- and multi-target)
        if is_multi_target:
            all_labels = sorted(
                set(t for targets in df["target_label_names"] for t in targets)
            )

            df["label"] = df["target_label_names"].apply(
                lambda targets: [t for t in targets]
            )
        else:
            all_labels = sorted(set(target["name"] for target in df["target"]))

            # set the label col
            df["label"] = df["target"].apply(lambda t: t["name"])

        label_metrics = {}
        for label in all_labels:
            label_metrics[str(label)] = {
                mp.metric_name: mp.summarize(
                    df,
                    is_multi_target=is_multi_target,
                    label=label,
                )
                for mp in metric_post_processors
            }
            logger.info(
                f"Summarized metrics for label '{label}': {label_metrics[str(label)]}"
            )

        dataset_level_metrics_path = run_dir / "dataset_level_metrics.json"
        payload = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "explainer": explainer_type,
            "n_samples": len(df),
            "metrics": _format_metrics_for_logging(summarized_metrics),
            "label_metrics": {
                label: _format_metrics_for_logging(lm)
                for label, lm in label_metrics.items()
            },
        }

        with open(dataset_level_metrics_path, "w") as f:
            json.dump(payload, f, indent=4)

        with open(dataset_level_metrics_path, "r") as f:
            return json.load(f)

    def summarize_explanations(self, force_recompute: bool = False):
        # prepare the summary output dir
        explanation_summary_dir = self._run_dir
        explanation_summary_dir.mkdir(parents=True, exist_ok=True)

        # set a image dir at the root as its same for all explanations
        sample_summary_dir = self._run_dir.parent
        sample_summary_dir.mkdir(parents=True, exist_ok=True)

        dataset_level_metrics_path = self._run_dir / "dataset_level_metrics.json"
        # if dataset_level_metrics_path.exists() and not force_recompute:
        #     return
        _ = self.state  # ensure state is initialized before accessing cacher
        sample_keys = self._explanations_cacher.list_sample_keys()
        logger.info(f"Summarizing explanations for {len(sample_keys)} samples...")

        # self._run_and_cache_predictions(sample_keys=sample_keys)

        # batch_size = 4
        # dataloader = DataLoader(
        #     SummaryBuilderDataset(
        #         analyzer=self,
        #         keys=sample_keys,
        #         sample_summary_root=sample_summary_dir,
        #         explanation_summary_root=explanation_summary_dir,
        #         load_explanations=True,
        #     ),
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=8,
        #     collate_fn=lambda batch: batch,
        # )

        # prepare sample level summaries
        # for batch in tqdm.tqdm(dataloader, desc="Generating summaries"):
        #     pass

        # now we load the summaries and prepare the dataset level metrics summary
        metadata_list = []
        for sample_id in tqdm.tqdm(
            sample_keys, desc="Loading summaries for dataset-level metrics"
        ):
            try:
                metadata = ExplanationSummary.load_metadata_from_disk(
                    sample_id=sample_id,
                    root_dir=explanation_summary_dir,
                )
            except KeyError as e:
                logger.error(
                    f"Error loading explanation summary metadata for sample {sample_id}: {e}"
                )
                continue
            explanation_metrics = metadata.pop("explanation_metrics", {})

            # convert metric lists to np arrays
            for metric_name, metric_value in explanation_metrics.items():
                if isinstance(metric_value, list):
                    explanation_metrics[metric_name] = np.array(metric_value)

            diagnostics_metrics = metadata.pop("diagnostics_metrics", {})

            for diag_name, diag_value in diagnostics_metrics.items():
                if isinstance(diag_value, list):
                    diagnostics_metrics[diag_name] = np.array(diag_value)

            metadata_list.append(
                {
                    **metadata,
                    **explanation_metrics,
                    **diagnostics_metrics,
                }
            )
        df = pd.DataFrame.from_records(metadata_list)
        logger.info(f"Dataset explanation metadata and metrics Dataframe:\n{df}")
        self.compute_and_store_dataset_metrics(
            df=df,
            run_dir=self._run_dir,
            dataset_name=self._config.env.dataset_name,
            model_name=self._config.env.model_name,
            explainer_type=self._config.x_model_pipeline.explainer.type,
            metric_post_processors=METRIC_POST_PROCESSORS,
        )

    @torch.no_grad()
    def visualize_random_batch(self, batch_size: int = 2, save_to_disk: bool = True):
        # get all the sample ids from the explanations cacher
        _ = self.state  # ensure state is initialized before accessing cacher

        logger.info(
            "Loading sample keys from explanations cacher from file: {}".format(
                self._explanations_cacher.file_path
            )
        )
        sample_keys = self._explanations_cacher.list_sample_keys()

        # show sample_keys
        logger.info(
            f"Available sample keys for visualization: {len(sample_keys)} samples found."
        )

        # get random sample ids
        sample_ids = random.sample(sample_keys, k=batch_size)
        sample_ids_hash = hashlib.md5("_".join(sample_ids).encode()).hexdigest()[:8]
        viz_dir = self._run_dir.parent.parent / self._run_dir.name

        if save_to_disk:
            viz_dir.mkdir(parents=True, exist_ok=True)
            file_path = viz_dir / f"batch-explanation-{sample_ids_hash}.png"
            # print("file_path", file_path.exists())
            # if file_path.exists():
            #     logger.info(
            #         f"Batch explanation visualization already exists at {file_path}. Skipping visualization."
            #     )
            #     return

            logger.info(f"Visualizing explanations for sample IDs: {sample_ids}")
            explanation_summaries = []
            for sample_id in sample_ids:
                explanation_summary = self.get_explanation_summarizer(
                    sample_id=sample_id
                )
                explanation_summaries.append(explanation_summary)

            print("explanation_summary", explanation_summary)
            # # save the explanation tensors to the disk
            # for summary in explanation_summaries:
            #     with open(
            #         viz_dir / f"{summary.sample.sample_id}_explanation_tensors.pt", "wb"
            #     ) as f:
            #         pickle.dump(summary, f)

            fig = ExplanationSummarizer.visualize_batch(
                explanation_summaries=explanation_summaries,
            )

            logger.info(
                f"Saving batch explanation visualization to {viz_dir / f'batch-explanation-{sample_ids_hash}.png'}"
            )
            fig.savefig(
                viz_dir / f"batch-explanation-{sample_ids_hash}.png",
                bbox_inches="tight",
                dpi=120,
            )
        else:
            logger.info(f"Visualizing explanations for sample IDs: {sample_ids}")
            explanation_summaries = []
            for sample_id in sample_ids:
                explanation_summary = self.get_explanation_summarizer(
                    sample_id=sample_id
                )
                explanation_summaries.append(explanation_summary)

            fig = ExplanationSummarizer.visualize_batch(
                explanation_summaries=explanation_summaries,
            )
            fig.show()

    @torch.no_grad()
    def visualize_all_samples(self):
        # get all the sample ids from the explanations cacher
        sample_keys = self._explanations_cacher.list_sample_keys()

        viz_dir = Path(self._run_dir) / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        for sample_id in tqdm.tqdm(sample_keys, desc="Visualizing samples"):
            explanation_summary = self.get_explanation_summarizer(sample_id=sample_id)
            explanation_summary.visualize_sample(
                sample_id=sample_id,
                save_to_disk=True,
                save_path=viz_dir / f"sample-{sample_id}.png",
            )

    @torch.no_grad()
    def diagnose_random_batch(self, batch_size: int = 4, save_to_disk: bool = True):
        import matplotlib.pyplot as plt

        # get all the sample ids from the explanations cacher
        logger.info(
            "Loading sample keys from explanations cacher from file: {}".format(
                self._explanations_cacher.file_path
            )
        )
        sample_keys = self._explanations_cacher.list_sample_keys()

        # show sample_keys
        logger.info(
            f"Available sample keys for visualization: {len(sample_keys)} samples found."
        )

        # get random sample ids
        sample_ids = random.sample(sample_keys, k=batch_size)
        sample_ids_hash = hashlib.md5("_".join(sample_ids).encode()).hexdigest()[:8]
        if save_to_disk:
            diagnostics_dir = Path(
                self._run_dir / "viz" / f"diagnostics-{sample_ids_hash}"
            )
            # if diagnostics_dir.exists():
            #     logger.info(
            #         f"Diagnostics already exist at {diagnostics_dir}. Skipping."
            #     )
            #     return

            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Running modality diagnostics for sample IDs: {sample_ids}")
            all_sample_stats = []
            for sample_id in sample_ids:
                explanation_summary = self.get_explanation_summarizer(
                    sample_id=sample_id
                )
                safe_id = str(sample_id).replace("/", "_")

                fig_detail, fig_summary, sample_stats = plot_modality_diagnostics(
                    explanation_summary, show_pre_reduction=True
                )

                for s in sample_stats:
                    s["sample_id"] = sample_id
                all_sample_stats.extend(sample_stats)

                fig_detail.savefig(
                    diagnostics_dir / f"{safe_id}_detail.png",
                    bbox_inches="tight",
                    dpi=120,
                )
                fig_summary.savefig(
                    diagnostics_dir / f"{safe_id}_summary.png",
                    bbox_inches="tight",
                    dpi=120,
                )
                plt.close(fig_detail)
                plt.close(fig_summary)

            # write aggregated stats CSV
            if all_sample_stats:
                import csv

                csv_path = diagnostics_dir / "modality_stats.csv"
                fieldnames = list(all_sample_stats[0].keys())
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_sample_stats)

            logger.info(f"Saved modality diagnostics to {diagnostics_dir}")
        else:
            logger.info(f"Running modality diagnostics for sample IDs: {sample_ids}")
            all_sample_stats = []
            for sample_id in sample_ids:
                explanation_summary = self.get_explanation_summarizer(
                    sample_id=sample_id
                )

                fig_detail, fig_summary, sample_stats = plot_modality_diagnostics(
                    explanation_summary, show_pre_reduction=True
                )

                for s in sample_stats:
                    s["sample_id"] = sample_id
                all_sample_stats.extend(sample_stats)

                fig_detail.show()
                fig_summary.show()

    def _aggregate_summarized_metrics(
        self,
        base_dir: Path | None = None,
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        explainers: list[str] | None = None,
        exclude_metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate summarized_explanation_metrics.json files into a DataFrame.

        Parameters
        ----------
        base_dir : Path, optional
            Root directory to glob under. Defaults to self._explanations_run_dir.
        datasets : list[str], optional
            Keep only these dataset names (raw, as stored in the JSON).
        models : list[str], optional
            Keep only these model names.
        explainers : list[str], optional
            Keep only these explainer folder names.
        exclude_metrics : list[str], optional
            Metric names to skip. Defaults to ['AOPC'].

        Returns
        -------
        pd.DataFrame
            One row per (dataset, model, explainer, metric) combination.
        """
        import glob

        if base_dir is None:
            base_dir = self._explanations_run_dir
        if exclude_metrics is None:
            exclude_metrics = ["AOPC"]

        pattern = str(Path(base_dir) / "**" / "summarized_explanation_metrics.json")
        files = glob.glob(pattern, recursive=True)

        if not files:
            logger.warning(
                f"No summarized_explanation_metrics.json files found under {base_dir}. "
                "Run mode='prepare_metrics' first."
            )
            return pd.DataFrame()

        logger.info(f"Found {len(files)} summarized metric files under {base_dir}")

        agg_metric_data = []
        for file in files:
            path = Path(file)
            explainer_name = path.parent.name

            with open(file, "r") as f:
                data = json.load(f)

            dataset_name = data.get("dataset_name", self._config.env.dataset_name)
            model_name = data.get("model_name", self._config.env.model_name)

            # Apply filters
            if datasets and dataset_name not in datasets:
                continue
            if models and model_name not in models:
                continue
            if explainers and explainer_name not in explainers:
                continue

            for metric_name, metric_value in data.get("metrics", {}).items():
                if metric_name in exclude_metrics:
                    continue
                agg_metric_data.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "explainer": explainer_name,
                        "metric_name": metric_name,
                        "metric_value": metric_value["score"],
                        "exec_time": metric_value.get("exec_time"),
                        "metric_type": metric_value["type"],
                        "is_lower_the_better": metric_value.get("is_lower_the_better"),
                        "metric_perturbation_type": metric_value.get(
                            "metric_perturbation_type"
                        ),
                    }
                )

        df = pd.DataFrame.from_records(agg_metric_data)
        if not df.empty:
            logger.info(
                f"Aggregated {len(df)} metric records across "
                f"{df['explainer'].nunique()} explainer(s)"
            )
        return df

    def generate_metric_corrs(self):
        """Generate metric correlation matrices for faithfulness and complexity metrics."""
        corr_output_dir = self._run_dir / "metric_correlations"
        corr_output_dir.mkdir(parents=True, exist_ok=True)

        df = self._aggregate_summarized_metrics()
        if df.empty:
            return

        for metric_type, group in df.groupby("metric_type"):
            if metric_type not in ("complexity", "faithfulness"):
                continue

            pivot = group.pivot_table(
                index=["dataset", "model", "explainer"],
                columns="metric_name",
                values="metric_value",
            )

            if pivot.shape[1] < 2:
                logger.info(
                    f"Skipping correlation for '{metric_type}': only {pivot.shape[1]} metric(s)."
                )
                continue

            corr_matrix = pivot.corr()

            # Save to run dir
            csv_path = corr_output_dir / f"corr_matrix_{metric_type}.csv"
            corr_matrix.to_csv(csv_path)
            logger.info(f"Saved correlation matrix CSV: {csv_path}")

    def run(self):
        if self._config.mode == "prepare_metrics":
            self.summarize_metrics(force_recompute=self._config.force_recompute)
        elif self._config.mode == "summarize_explanations":
            self.summarize_explanations(force_recompute=self._config.force_recompute)
        elif self._config.mode == "batch":
            for _ in range(self._config.n_batches):
                self.visualize_random_batch(self._config.batch_size)
        elif self._config.mode == "diagnose":
            for _ in range(self._config.n_batches):
                self.diagnose_random_batch()
        elif self._config.mode == "save":
            self.visualize_all_samples()
        elif self._config.mode == "metric_corrs":
            self.generate_metric_corrs()
        elif self._config.mode == "expl_analysis":
            self.run_expl_analysis()
        else:
            raise ValueError(f"Mode {self._config.mode} not recognized.")
