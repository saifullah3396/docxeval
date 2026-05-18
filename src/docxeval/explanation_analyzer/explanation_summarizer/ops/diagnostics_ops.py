from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from atria_logger._api import get_logger
from atria_types._generic._annotations import AnnotationType
from atria_types._generic._bounding_box import BoundingBox
from torchxai.metrics.diagnosis.attribution_locality import attribution_locality
from torchxai.metrics.diagnosis.attribution_text_analysis import (
    attribution_text_analysis,
)
from torchxai.metrics.diagnosis.modality_topk_fraction import modality_topk_fraction

if TYPE_CHECKING:
    from docxeval.explanation_analyzer.explanation_summarizer.explanation_summarizer import (
        ExplanationSummarizer,
    )

logger = get_logger(__name__)


class ExplanationDiagnosticsOps:
    def __init__(
        self,
        summarizer: "ExplanationSummarizer",
        k_fractions: list[float] | None = None,
    ) -> None:
        self.s = summarizer
        self.k_fractions = k_fractions or [0.1]
        # self.tokenizer, self.bert_model = self._load_bert_model()

    # def _load_bert_model(self):
    #     from transformers import AutoModel, AutoTokenizer

    #     model_name = "bert-base-uncased"
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.bert_model = AutoModel.from_pretrained(model_name)
    #     self.bert_model.eval()
    #     return self.tokenizer, self.bert_model

    # def _get_bert_embeddings(
    #     self,
    #     words: list[str],
    # ) -> torch.Tensor:
    #     embeddings = []
    #     with torch.no_grad():
    #         for word in words:
    #             inputs = self.tokenizer(word, return_tensors="pt")
    #             outputs = self.bert_model(**inputs)
    #             # mean pool over subword tokens, squeeze batch dim
    #             emb = outputs.last_hidden_state[0].mean(dim=0)
    #             embeddings.append(emb)

    #     return torch.stack(embeddings)  # [G, hidden_size]

    def _get_modality_topk_fraction(self):
        # we run diagnosis on reduced explanations
        # for multimodal it will be [1, num_features] per modality; for multi-target it will be list of such tensors
        # and for image it will be [1, 14, 14]
        explanations = self.s.reduced_explanations()
        return modality_topk_fraction(
            attributions=explanations,
            modality_names=self.s.explanation_state.feature_keys,
            k_fractions=self.k_fractions,
            multi_target=self.s._explanation_mode == "multi",
            return_dict=True,
        )

    def _get_word_attribution_locality(self):
        # for attribution locality we only consider the token-level aggregate explanations
        explanations = self.s.reduced_explanations()

        # generate aggregated explanations
        word_level_explanations = []
        for explanation_per_target in explanations:
            # extract all attributions per token and their corresponding bboxes, then call the locality function
            word_level_explanation = []
            for feature_key in self.s.explanation_state.feature_keys:
                if feature_key not in [
                    "token_embeddings",
                    "position_embeddings",
                    "layout_embeddings",
                ]:
                    continue
                idx = self.s.explanation_state.feature_keys.index(feature_key)
                word_level_explanation.append(explanation_per_target[idx])
            aggregated_explanation = torch.stack(word_level_explanation, dim=0).sum(
                dim=0
            )

            word_level_explanations.append(aggregated_explanation)

        words, bboxes, _ = self.s.get_words_and_bboxes()
        if self.s.transformed_sample.metadata.qa_question:
            context = self.s.transformed_sample.metadata.qa_question
            words = context.split() + words
            bboxes = [BoundingBox(value=[0, 0, 0, 0])] * len(context.split()) + bboxes

        word_bboxes = torch.tensor([x.value for x in bboxes]).unsqueeze(
            0
        )  # add batch dim

        # word explanations are of shape [batch size, num_words]
        assert (
            word_level_explanations[0].shape[1] == word_bboxes.shape[1]
        ), "Number of tokens in explanations should match number of bboxes"
        return attribution_locality(
            attributions=word_level_explanations,
            target_indices=list(
                range(len(word_level_explanations))
            ),  # our explanations correspond to the words
            feature_mask=None,
            bboxes=word_bboxes,
            use_weighted_sum=False,
            return_dict=True,
        )

    def _get_text_profile(self):
        # for attribution locality we only consider the token-level aggregate explanations
        explanations = self.s.reduced_explanations()

        if isinstance(explanations, list):
            # generate aggregated explanations
            word_level_explanations = []
            for explanation_per_target in explanations:
                word_level_explanation = []
                for feature_key in self.s.explanation_state.feature_keys:
                    if feature_key not in [
                        "token_embeddings",
                        "position_embeddings",
                        "layout_embeddings",
                    ]:
                        continue
                    idx = self.s.explanation_state.feature_keys.index(feature_key)
                    word_level_explanation.append(explanation_per_target[idx])
                aggregated_explanation = torch.stack(word_level_explanation, dim=0).sum(
                    dim=0
                )

                word_level_explanations.append(aggregated_explanation)

            word_labels = None
            if self.s.sample.has_annotation_type(AnnotationType.entity_labeling):
                annotation = self.s.sample.get_annotation_by_type(
                    annotation_type=AnnotationType.entity_labeling
                )
                word_labels = annotation.word_labels

            words, bboxes, word_ids = self.s.get_words_and_bboxes()
            if self.s.transformed_sample.metadata.qa_question:
                context = self.s.transformed_sample.metadata.qa_question
                words = context.split() + words
                bboxes = [BoundingBox(value=[0, 0, 0, 0])] * len(
                    context.split()
                ) + bboxes

            if word_labels is not None:  # for ner
                token_labels = [word_labels[word_id].name for word_id in word_ids]
                target_indices = list(range(len(word_level_explanations)))
            else:  # for qa
                target_indices = self.s.qa_targets
                if (
                    target_indices[0] < 0
                    or target_indices[1] < 0
                    or target_indices[0] >= len(words)
                    or target_indices[1] >= len(words)
                ):
                    target_indices = None

            # word explanations are of shape [batch size, num_words]
            assert word_level_explanations[0].shape[1] == len(
                words
            ), "Number of tokens in explanations should match number of words"
            return attribution_text_analysis(
                attributions=word_level_explanations,
                tokens=[words],  # add batch dim
                token_labels=(
                    [token_labels] if word_labels is not None else None
                ),  # add batch dim
                # token_embeddings=[self._get_bert_embeddings(words)],  # add batch dim
                target_indices=target_indices,  # our explanations correspond to the words
                use_weighted_sum=False,
            )
        else:
            word_level_explanation = []
            for feature_key in self.s.explanation_state.feature_keys:
                if feature_key not in [
                    "token_embeddings",
                    "position_embeddings",
                    "layout_embeddings",
                ]:
                    continue
                idx = self.s.explanation_state.feature_keys.index(feature_key)
                word_level_explanation.append(explanations[idx])
            word_level_explanation = torch.stack(word_level_explanation, dim=0).sum(
                dim=0
            )

            words, bboxes, _ = self.s.get_words_and_bboxes()

            # word explanations are of shape [batch size, num_words]
            assert word_level_explanation.shape[1] == len(
                words
            ), "Number of tokens in explanations should match number of words"
            return attribution_text_analysis(
                attributions=[word_level_explanation],
                tokens=[words],  # add batch dim
                token_labels=None,
                # token_embeddings=[self._get_bert_embeddings(words)],  # add batch dim
                target_indices=None,  # our single explanation corresponds to the words
                use_weighted_sum=False,
            )
