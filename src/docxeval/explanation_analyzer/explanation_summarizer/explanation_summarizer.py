from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from atria_insights.data_types._explanation_state import (
    MultiTargetSampleExplanation,
    SampleExplanation,
    SampleExplanationState,
    SampleExplanationTarget,
)
from atria_insights.data_types._metric_data import SampleMetricData
from atria_logger._api import get_logger
from atria_models.core.types.model_outputs import ModelOutput
from atria_transforms.data_types._document import DocumentTensorDataModel
from atria_transforms.data_types._image import ImageTensorDataModel
from atria_types._data_instance._document_instance import DocumentInstance
from atria_types._data_instance._image_instance import ImageInstance
from atria_types._generic._annotations import AnnotationType
from atria_types._generic._bounding_box import BoundingBox

from docxeval.explanation_analyzer.explanation_summarizer.explanation_summary import (
    ExplanationSummary,
    SampleSummary,
    _numpy_to_item_recursive,
    _tensor_to_numpy_recursive,
)
from docxeval.explanation_analyzer.explanation_summarizer.ops.diagnostics_ops import (
    ExplanationDiagnosticsOps,
)
from docxeval.explanation_analyzer.explanation_summarizer.ops.reduce_ops import (
    ExplanationReductionOps,
)
from docxeval.explanation_analyzer.explanation_summarizer.ops.viz_ops import (
    ExplanationVisualizationOps,
)
from docxeval.explanation_analyzer.utils.explanation_units import (
    AggregateTextExplanationUnit,
    ImageExplanationUnit,
    TextExplanationUnit,
    TextLayoutExplanationUnit,
    TextPositionExplanationUnit,
)

logger = get_logger(__name__)


class ExplanationSummarizer:
    def __init__(
        self,
        sample: DocumentInstance | ImageInstance,
        transformed_sample: DocumentTensorDataModel | ImageTensorDataModel,
        model_output: ModelOutput,
        explanation_state: SampleExplanationState,
        explanation_metrics: dict[str, SampleMetricData],
        dataset_name: str | None = None,
        model_name: str | None = None,
        explainer: str | None = None,
        dataset_labels: dict[str, Any] | None = None,
        qa_targets: list[int] | None = None,
    ):
        self.sample = sample
        self.transformed_sample = transformed_sample
        self.model_output = model_output
        self.explanation_state = explanation_state
        self.explanation_metrics = explanation_metrics
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.explainer = explainer
        self.reduction_ops = ExplanationReductionOps(self)
        self.visualization_ops = ExplanationVisualizationOps(self)
        self.diagnostic_ops = ExplanationDiagnosticsOps(self)
        self.dataset_labels = dataset_labels
        self.qa_targets = qa_targets

    def save_explanation_tensors_to_disk(self, directory: str):
        self.reduction_ops.save_explanation_tensors_to_disk(directory)

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    @property
    def _explanation_mode(self) -> str:
        """Return 'multi' when explanations is MultiTargetSampleExplanation, else 'single'."""
        if isinstance(
            self.explanation_state.explanations, MultiTargetSampleExplanation
        ):
            return "multi"
        return "single"

    def get_words_and_bboxes(self) -> tuple[list[str], list[torch.Tensor]]:
        text = self.sample.content.text_list
        bboxes = self.sample.content.bbox_list

        # get the words and bboxes in the current sample corresponding to the word ids
        if len(text) == 0:
            text = ["dummy"]
            bboxes = [BoundingBox(value=[0, 0, 0, 0], normalized=True)]
            word_ids = [0]
        else:
            text = []
            bboxes = []
            word_ids = []
            last_word_id = None

            # see if sequence_id > 0 max
            max_seq_id_is_greater_than_0 = max(self.transformed_sample.sequence_ids) > 0

            for sequence_id, word_id in zip(
                self.transformed_sample.sequence_ids, self.transformed_sample.word_ids
            ):
                if max_seq_id_is_greater_than_0 and sequence_id == 0:
                    continue  # skip special tokens in the first sequence
                if word_id == -100 or word_id == last_word_id:
                    continue
                text.append(self.sample.content.text_list[word_id])
                bboxes.append(self.sample.content.bbox_list[word_id])
                word_ids.append(word_id)
                last_word_id = word_id
        return text, bboxes, word_ids

    def _iter_target_explanations(
        self,
    ) -> list[tuple[int, SampleExplanationTarget | None, SampleExplanation]]:
        """Return a list of (target_idx, target, SampleExplanation) for each target.

        Single-target mode yields one element; multi-target mode yields one per target.
        """
        if self._explanation_mode == "multi":
            assert isinstance(
                self.explanation_state.explanations, MultiTargetSampleExplanation
            )
            targets = (
                self.explanation_state.target
                if isinstance(self.explanation_state.target, list)
                else [None] * self.explanation_state.explanations.n_targets
            )
            return [
                (idx, target, sample_expl)
                for idx, (target, sample_expl) in enumerate(
                    zip(targets, self.explanation_state.explanations.value, strict=True)
                )
            ]
        else:
            assert isinstance(self.explanation_state.explanations, SampleExplanation)
            target = (
                self.explanation_state.target
                if not isinstance(self.explanation_state.target, list)
                else None
            )
            return [(0, target, self.explanation_state.explanations)]

    # ------------------------------------------------------------------
    # Single-target pure pipeline
    # ------------------------------------------------------------------

    def _reduced_explanations_single(
        self, sample_explanation: SampleExplanation
    ) -> tuple[torch.Tensor, ...]:
        return self.reduction_ops._reduced_explanations_single(sample_explanation)

    def _explanation_units_single(
        self,
        normalized: dict[str, torch.Tensor],
        normalized_with_agg_dict: dict[str, torch.Tensor],
        raw_sample_explanation: SampleExplanation,
        target: SampleExplanationTarget | None = None,
    ) -> list:
        """Build the list of ExplanationUnit objects for a single normalized target."""
        units = []

        text, bboxes, _ = self.get_words_and_bboxes()

        target_label = None
        target_word_bbox = None
        target_text = None
        predicted_answer = None
        if target is not None and self.dataset_labels.ser is not None:
            dataset_labels = self.dataset_labels.ser
            token_labels = self.transformed_sample.token_labels
            # now go over targets and extract the label of each
            label_idx = token_labels[target.value]
            target_label = dataset_labels[label_idx]
            target = self.transformed_sample.word_ids[target.value]
            target_text = self.sample.content.text_list[target]
            target_word_bbox = self.sample.content.bbox_list[target]

        # see if this is a question answering sample by checking the annotation
        context_text = None
        context_attributions = {}
        context_attributions_agg = {}
        if self.transformed_sample.metadata.qa_question is not None:
            context_text = self.transformed_sample.metadata.qa_question.split()
            predicted_answer = self.model_output.answer[0]
            for key, value in normalized.items():
                if key in [
                    "token_embeddings",
                    "position_embeddings",
                    "layout_embeddings",
                ]:
                    normalized[key] = value[:, len(context_text) :]
                    context_attributions[key] = value[:, : len(context_text)]
            context_attributions_agg["agg"] = normalized_with_agg_dict["agg"][
                :, : len(context_text)
            ]
            normalized_with_agg_dict["agg"] = normalized_with_agg_dict["agg"][
                :, len(context_text) :
            ]

            # for question answer we can also extract the target_word_bbox from the answer start and end indices
            # answer_start = self.transformed_sample.metadata.
            # target_word_bbox

        if self.explainer not in ["attn/raw_attention", "attn/attention_rollout"]:
            units.append(
                TextExplanationUnit(
                    text=text,
                    context_text=context_text,
                    predicted_answer=predicted_answer,
                    bboxes=bboxes,
                    attribution=normalized["token_embeddings"][0].numpy(),
                    context_attribution=(
                        context_attributions["token_embeddings"][0].numpy()
                        if context_text is not None
                        else None
                    ),
                    target_text=target_text,
                    target_word_bbox=target_word_bbox,
                    target_label=target_label,
                )
            )

            if "position_embeddings" in self.explanation_state.feature_keys:
                units.append(
                    TextPositionExplanationUnit(
                        text=text,
                        context_text=context_text,
                        predicted_answer=predicted_answer,
                        bboxes=bboxes,
                        attribution=normalized["position_embeddings"][0].numpy(),
                        context_attribution=(
                            context_attributions["position_embeddings"][0].numpy()
                            if context_text is not None
                            else None
                        ),
                        target_text=target_text,
                        target_word_bbox=target_word_bbox,
                        target_label=target_label,
                    )
                )

            if "layout_embeddings" in self.explanation_state.feature_keys:
                units.append(
                    TextLayoutExplanationUnit(
                        text=text,
                        context_text=context_text,
                        predicted_answer=predicted_answer,
                        bboxes=bboxes,
                        attribution=normalized["layout_embeddings"][0].numpy(),
                        context_attribution=(
                            context_attributions["layout_embeddings"][0].numpy()
                            if context_text is not None
                            else None
                        ),
                        target_text=target_text,
                        target_word_bbox=target_word_bbox,
                        target_label=target_label,
                    )
                )

        # only add if not raw attn
        units.append(
            AggregateTextExplanationUnit(
                text=text,
                context_text=context_text,
                predicted_answer=predicted_answer,
                bboxes=bboxes,
                attribution=normalized_with_agg_dict["agg"][0].numpy(),
                context_attribution=(
                    context_attributions_agg["agg"][0].numpy()
                    if context_text is not None
                    else None
                ),
                target_text=target_text,
                target_word_bbox=target_word_bbox,
                target_label=target_label,
            )
        )

        if "image" in self.explanation_state.feature_keys:
            units.append(
                ImageExplanationUnit(
                    attribution=normalized["image"][0].numpy(),
                    target_text=target_text,
                    target_word_bbox=target_word_bbox,
                    target_label=target_label,
                )
            )

        return units

    def generate_sample_summary(self) -> SampleSummary:
        sample_image = self.sample.image.content

        # lets divide sample metadata into top level and details for better storage and reading
        # also lets add model outputs as inference results
        metadata = self.sample.to_row(
            exclude={
                "image": {"content"},
                "ocr": {"content"},
                "content": {"text", "text_elements"},
            }
        )

        # check if classification annotation exists and add predicted and true labels to metadata
        # lets also add it to the top level for easier access
        if self.sample.has_annotation_type(AnnotationType.classification):
            classification_annotation = self.sample.get_annotation_by_type(
                AnnotationType.classification
            )
            metadata["true_label"] = classification_annotation.label.name
            metadata["predicted_label"] = self.model_output.predicted_label_name[0]
        else:
            metadata["true_label"] = None
            metadata["predicted_label"] = None

        metadata["content"] = self.sample.model_dump(include={"content"})["content"]

        # lets also add model output to metadata for easier access
        metadata["model_output"] = self.model_output.to_dict()

        return SampleSummary(
            sample_id=self.sample.sample_id,
            metadata=metadata,
            image=sample_image,
        )

    def generate_explanation_summary(self) -> ExplanationSummary:
        units = self.explanation_units()
        serialized_units: Any
        if self._explanation_mode == "multi":
            # list of dicts, one per target, ordered by target index
            multi_units = cast(list[list[Any]], units)
            serialized_units = []
            for target_units in multi_units:
                target_units_typed = cast(list[Any], target_units)
                serialized_units.append(
                    {unit.name: unit.model_dump() for unit in target_units_typed}
                )
        else:
            single_units = cast(list[Any], units)
            serialized_units = {unit.name: unit.model_dump() for unit in single_units}
        return ExplanationSummary(
            sample_id=self.sample.sample_id,
            metadata=self.get_explanation_metadata(),
            explanation_units=serialized_units,
        )

    def get_normalized_metrics(self) -> dict[str, float]:
        normalized_metrics = {"explanation_metrics": {}, "diagnosis_metrics": {}}
        for metric_name, metric_value in self.explanation_metrics.items():
            _, metric_name = metric_name.split("/")
            for sub_metric_name, sub_metric_value in metric_value.data.items():
                normalized_metrics["explanation_metrics"][
                    f"{metric_name}.{sub_metric_name}"
                ] = _numpy_to_item_recursive(
                    _tensor_to_numpy_recursive(sub_metric_value)
                )

        for metric_name, metric_value in self.get_diagnosis_metrics().items():
            normalized_metrics["diagnosis_metrics"][metric_name] = (
                _numpy_to_item_recursive(_tensor_to_numpy_recursive(metric_value))
            )

        return normalized_metrics

    def get_diagnosis_metrics(self):
        metrics = {}
        metrics["modality_topk_fraction"] = (
            self.diagnostic_ops._get_modality_topk_fraction()
        )
        metrics["text_profile"] = self.diagnostic_ops._get_text_profile()

        is_multi = self._explanation_mode == "multi"
        if is_multi:
            metrics["word_attribution_locality"] = (
                self.diagnostic_ops._get_word_attribution_locality()
            )

        return metrics

    def get_explanation_metadata(self) -> dict[str, Any]:
        is_correct = None
        if hasattr(self.model_output, "is_correct"):
            is_correct = self.model_output.is_correct()
            assert len(is_correct) == 1, "Expected batch size of 1 for is_correct"
            is_correct = is_correct[0]

        if isinstance(self.explanation_state.target, list):
            is_correct_per_target = []
            for target in self.explanation_state.target:
                assert isinstance(
                    target, SampleExplanationTarget
                ), "Expected target to be of type SampleExplanationTarget"
                is_correct_per_target.append(
                    is_correct[target.value]
                    if isinstance(is_correct, list)
                    else is_correct
                )

            assert len(self.explanation_state.target) == len(
                is_correct_per_target
            ), "Length of explanation_state.target should match length of is_correct"
            is_correct = is_correct_per_target

        is_multi = self._explanation_mode == "multi"
        targets_list = (
            self.explanation_state.target
            if isinstance(self.explanation_state.target, list)
            else None
        )
        single_target = (
            self.explanation_state.target
            if not isinstance(self.explanation_state.target, list)
            else None
        )
        n_targets = (
            len(self.explanation_state.explanations.value)
            if isinstance(
                self.explanation_state.explanations, MultiTargetSampleExplanation
            )
            else 1
        )

        # for entity type labels we must add the label name to the metadata for better interpretability
        if is_multi and targets_list is not None:
            dataset_labels = self.dataset_labels.ser
            if dataset_labels is not None:
                token_labels = self.transformed_sample.token_labels
                # now go over targets and extract the label of each
                target_labels = []
                for target in targets_list:
                    label_idx = token_labels[target.value]
                    label_value = dataset_labels[label_idx]
                    target_labels.append(label_value)
            else:
                target_labels = [f"target_{target.name}" for target in targets_list]

        return {
            "sample_id": self.sample.sample_id,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "explainer": self.explainer,
            "is_correct": is_correct,
            "explanation_type": "multi" if is_multi else "single",
            "n_targets": n_targets,
            "target": single_target,
            "targets": targets_list,
            "target_label_names": target_labels if is_multi else None,
            "feature_keys": self.explanation_state.feature_keys,
            "frozen_features": self.explanation_state.frozen_features,
            "sliding_window_shapes": self.explanation_state.sliding_window_shapes,
            "strides": self.explanation_state.strides,
            **self.get_normalized_metrics(),
        }

    def reduced_explanations(
        self,
    ) -> tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]:
        return self.reduction_ops.reduced_explanations()

    def normalize_explanations(
        self,
        explanations: tuple[torch.Tensor, ...],
        shift_0_to_1: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        return self.reduction_ops.normalize_explanations(
            explanations=explanations,
            shift_0_to_1=shift_0_to_1,
        )

    def visualize_explanation_distributions(
        self,
        explanations: dict[str, Any],
        title_prefix: str = "",
    ):
        self.visualization_ops.visualize_explanation_distributions(
            explanations=explanations,
            title_prefix=title_prefix,
        )

    def _explanation_units_for_single_expl(
        self,
        sample_expl: SampleExplanation,
        visualize_raw: bool = False,
        visualize_normalized: bool = False,
        target: Any = None,
    ) -> list:
        """Full pipeline for one SampleExplanation → list of ExplanationUnit."""
        reduced = self._reduced_explanations_single(sample_expl)
        reduced_with_keys = {
            key: value
            for key, value in zip(
                self.explanation_state.feature_keys, reduced, strict=True
            )
        }

        if visualize_raw:
            raw_dict = {
                key: value
                for key, value in zip(
                    self.explanation_state.feature_keys, reduced, strict=True
                )
            }
            self.visualize_explanation_distributions(raw_dict, title_prefix="Raw")

        normalized_tuple = self.normalize_explanations(reduced, shift_0_to_1=True)
        normalized = {
            key: value
            for key, value in zip(
                self.explanation_state.feature_keys, normalized_tuple, strict=True
            )
        }

        reduced_with_agg = sum(
            red
            for key, red in reduced_with_keys.items()
            if key in ["token_embeddings", "position_embeddings", "layout_embeddings"]
        )

        reduced_image = None
        if "image" in reduced_with_keys:
            reduced_image = reduced_with_keys["image"]

            normalized_with_agg = self.normalize_explanations(
                (reduced_with_agg, reduced_image), shift_0_to_1=True
            )
            normalized_with_agg_dict = {
                "agg": normalized_with_agg[0],
                "image": normalized_with_agg[1],
            }
        else:
            normalized_with_agg = self.normalize_explanations(
                (reduced_with_agg,), shift_0_to_1=True
            )
            normalized_with_agg_dict = {"agg": normalized_with_agg[0]}

        if visualize_normalized:
            self.visualize_explanation_distributions(
                normalized, title_prefix="Normalized"
            )

        return self._explanation_units_single(
            normalized, normalized_with_agg_dict, sample_expl, target=target
        )

    def explanation_units(
        self, visualize_raw: bool = False, visualize_normalized: bool = False
    ) -> list | list[list]:
        """Build explanation units.

        Single mode: returns list[ExplanationUnit].
        Multi mode: returns list[list[ExplanationUnit]] ordered by target index.
        """
        if self._explanation_mode == "multi":
            return [
                self._explanation_units_for_single_expl(
                    sample_expl,
                    visualize_raw=visualize_raw,
                    visualize_normalized=visualize_normalized,
                    target=target,
                )
                for _, target, sample_expl in self._iter_target_explanations()
            ]
        return self._explanation_units_for_single_expl(
            self._iter_target_explanations()[0][2],
            visualize_raw=visualize_raw,
            visualize_normalized=visualize_normalized,
        )

    def _visualize_unit_list(
        self,
        units: list,
        draw_title: bool = True,
        title_suffix: str = "",
    ) -> plt.Figure:
        return self.visualization_ops.visualize_unit_list(
            units=units,
            draw_title=draw_title,
            title_suffix=title_suffix,
        )

    def visualize_sample(
        self,
        draw_title: bool = True,
        return_fig: bool = False,
        target_idx: int | None = None,
    ):
        return self.visualization_ops.visualize_sample(
            draw_title=draw_title, return_fig=return_fig, target_idx=target_idx
        )

    def get_explanation_images(self) -> dict[str, PILImage] | list[dict[str, PILImage]]:
        return self.visualization_ops.get_explanation_images()

    @classmethod
    def visualize_batch(
        cls,
        explanation_summaries: list["ExplanationSummarizer"],
    ):
        return ExplanationVisualizationOps.visualize_batch(explanation_summaries)
