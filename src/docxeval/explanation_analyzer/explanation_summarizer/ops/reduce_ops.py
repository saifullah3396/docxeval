from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
from atria_insights.data_types._explanation_state import SampleExplanation
from atria_logger._api import get_logger

if TYPE_CHECKING:
    from docxeval.explanation_analyzer.explanation_summarizer.explanation_summarizer import (
        ExplanationSummarizer,
    )


logger = get_logger(__name__)


class ExplanationReductionOps:
    """Encapsulates reduction and normalization operations."""

    def __init__(self, summarizer: "ExplanationSummarizer") -> None:
        self.s = summarizer

    def _reduced_explanations_single(
        self, sample_explanation: SampleExplanation
    ) -> tuple[torch.Tensor, ...]:
        reduced: dict[str, torch.Tensor] = {}
        for key, explanation in zip(
            self.s.explanation_state.feature_keys,
            sample_explanation.value,
            strict=True,
        ):
            if key == "image":
                reduced["image"] = self.reduce_image(explanation).unsqueeze(0)
                continue

            word_level_explanations: dict = defaultdict(list)
            assert (
                explanation.shape[0] == 1
            ), "Expected batch size of 1 for explanations"
            for seq_id, word_id, explanation_per_token in zip(
                self.s.transformed_sample.sequence_ids,
                self.s.transformed_sample.word_ids,
                explanation[0],
            ):
                if word_id == -100:
                    continue
                word_key = (seq_id.item(), word_id.item())
                word_level_explanations[word_key].append(explanation_per_token)

            reduced_list = []
            for _, explanation_list in word_level_explanations.items():
                aggregated = torch.stack(explanation_list, dim=0)
                reduced_list.append(aggregated.sum() / len(explanation_list))
            reduced[key] = torch.stack(reduced_list, dim=0).unsqueeze(0)
        return tuple(reduced[k] for k in self.s.explanation_state.feature_keys)

    def reduce_image(self, image_attribution: torch.Tensor) -> torch.Tensor:
        # sum over channels: [1, C, H, W] -> [H, W]
        attr_2d = image_attribution.sum(dim=1).squeeze(0)

        # reduce over patches of cell size 16 x 16 (or whatever is configured) to get a smaller attribution map
        cell_size = 16

        # use torch unfold to creat epatches
        return (
            attr_2d.unfold(0, cell_size, cell_size)
            .unfold(1, cell_size, cell_size)
            .sum(-1)
            .sum(-1)
        )

    def reduce_image_to_word_patches(
        self,
        image_attribution: torch.Tensor,
        bboxes: list[list[float]],
        base_patch_size: int = 8,
        debug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attr_2d = image_attribution.sum(dim=1).squeeze(0)
        h, w = attr_2d.shape

        mask = torch.zeros(h, w, dtype=torch.long)
        patch_id = 1
        for row_start in range(0, h, base_patch_size):
            for col_start in range(0, w, base_patch_size):
                row_end = min(row_start + base_patch_size, h)
                col_end = min(col_start + base_patch_size, w)
                mask[row_start:row_end, col_start:col_end] = patch_id
                patch_id += 1

        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            px0 = max(0, min(round(x0 * w), w))
            py0 = max(0, min(round(y0 * h), h))
            px1 = max(0, min(round(x1 * w), w))
            py1 = max(0, min(round(y1 * h), h))

            if px0 == px1:
                px1 = min(px0 + 1, w)
            if py0 == py1:
                py1 = min(py0 + 1, h)

            mask[py0:py1, px0:px1] = patch_id
            patch_id += 1

        if debug:
            plt.figure(figsize=(6, 6))
            plt.imshow(mask, cmap="tab20")
            plt.title("Patch ID mask with word bboxes")
            plt.axis("off")
            plt.show()

        attributions = []
        for pid in torch.unique(mask):
            if pid == 0:
                continue
            region_mask = mask == pid
            attributions.append(attr_2d[region_mask].sum())

        if debug:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(self.s.sample.image.content.resize((w, h)))
            ax.imshow(mask.numpy(), cmap="tab20", alpha=0.5)
            ax.set_title("Patch mask overlaid on image")
            ax.axis("off")
            plt.show()

        return mask, torch.stack(attributions)

    def reduced_explanations(
        self,
    ) -> tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]:
        if self.s._explanation_mode == "multi":
            return [
                self._reduced_explanations_single(sample_expl)
                for _, _, sample_expl in self.s._iter_target_explanations()
            ]
        return self._reduced_explanations_single(
            self.s._iter_target_explanations()[0][2]
        )

    def normalize_explanations(
        self,
        explanations: tuple[torch.Tensor, ...],
        shift_0_to_1: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        from captum.attr._utils.visualization import _normalize_attr
        from torchxai.metrics._utils.common import (
            _split_tensors_to_tuple_tensors,
            _tuple_tensors_to_tensors,
        )

        explanations_flattened, original_shape = _tuple_tensors_to_tensors(
            tuple(x for x in explanations)
        )
        for idx in range(explanations_flattened.shape[0]):
            try:
                explanations_flattened[idx] = (
                    _normalize_attr(
                        explanations_flattened[idx].cpu(), sign="all", outlier_perc=1
                    )
                ).to(explanations_flattened.device)
            except Exception as e:
                logger.warning(
                    f"Normalization failed for explanation index {idx} with error: {e}. Skipping normalization for this explanation."
                )
                explanations_flattened[idx] = 0
            if shift_0_to_1:
                explanations_flattened[idx] = explanations_flattened[idx] * 0.5 + 0.5
        explanations_flattened = _split_tensors_to_tuple_tensors(
            explanations_flattened, original_shape
        )
        return explanations_flattened
