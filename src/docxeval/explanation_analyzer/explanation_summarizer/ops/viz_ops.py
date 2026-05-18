from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from atria_logger._api import get_logger
from atria_types._generic._annotations import AnnotationType
from matplotlib import gridspec
from PIL import ImageOps
from PIL.Image import Image as PILImage

if TYPE_CHECKING:
    from docxeval.explanation_analyzer.explanation_summarizer.explanation_summarizer import (
        ExplanationSummarizer,
    )
logger = get_logger(__name__)


class ExplanationVisualizationOps:
    """Encapsulates visualization operations for explanations."""

    def __init__(self, summarizer: "ExplanationSummarizer") -> None:
        self.s = summarizer

    def visualize_explanation_distributions(
        self,
        explanations: dict[str, Any],
        title_prefix: str = "",
    ) -> None:
        modality_names = list(explanations.keys())
        n_modalities = len(modality_names)

        fig, axes = plt.subplots(2, n_modalities, figsize=(5 * n_modalities, 8))
        if n_modalities == 1:
            axes = axes.reshape(2, 1)

        for col_idx, modality in enumerate(modality_names):
            attr_values = explanations[modality][0].cpu().numpy().flatten()

            axes[0, col_idx].bar(
                range(len(attr_values)), attr_values, alpha=0.7, color="blue"
            )
            title = f"{title_prefix} {modality}" if title_prefix else modality
            axes[0, col_idx].set_title(f"{title}\n(n={len(attr_values)})")
            axes[0, col_idx].set_xlabel("Index")
            axes[0, col_idx].set_ylabel("Attribution")
            axes[0, col_idx].grid(alpha=0.3)
            axes[0, col_idx].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            stats_text = f"μ={attr_values.mean():.3f}\nσ={attr_values.std():.3f}\nmin={attr_values.min():.3f}\nmax={attr_values.max():.3f}"
            axes[0, col_idx].text(
                0.02,
                0.98,
                stats_text,
                transform=axes[0, col_idx].transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            sorted_attrs = np.sort(attr_values)
            axes[1, col_idx].plot(sorted_attrs, linewidth=1)
            axes[1, col_idx].set_title("Sorted values")
            axes[1, col_idx].set_xlabel("Index (sorted)")
            axes[1, col_idx].set_ylabel("Attribution")
            axes[1, col_idx].grid(alpha=0.3)
            axes[1, col_idx].axhline(y=0, color="r", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

    def visualize_unit_list(
        self,
        units: list,
        draw_title: bool = True,
        title_suffix: str = "",
    ) -> plt.Figure:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(units),
            figsize=(5 * len(units), 5),
        )
        if len(units) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for axis_idx, explanation_unit in enumerate(units):
            image = self.s.sample.image.content

            explanation_image = explanation_unit.draw(image)

            expanded_image = ImageOps.expand(
                explanation_image, border=(0, 75, 0, 0), fill=(255, 255, 255)
            )
            axes[axis_idx].imshow(expanded_image)
            axes[axis_idx].axis("off")

            if draw_title:
                rectangle = patches.Rectangle(
                    (0, 1.0),
                    1.0,
                    -0.06,
                    transform=axes[axis_idx].transAxes,
                    color="gray",
                    alpha=0.25,
                )
                axes[axis_idx].add_patch(rectangle)
                label = explanation_unit.name
                if title_suffix:
                    label = f"{label} ({title_suffix})"
                axes[axis_idx].text(
                    0.5,
                    0.97,
                    label,
                    fontsize=14,
                    color="black",
                    ha="center",
                    va="center",
                    transform=axes[axis_idx].transAxes,
                )

        plt.tight_layout()
        return fig

    def visualize_sample(
        self,
        draw_title: bool = True,
        return_fig: bool = False,
        target_idx: int | None = None,
    ):
        units = self.s.explanation_units()
        if self.s._explanation_mode == "multi":
            figs = []
            for curr_target_idx, (_, target, _) in enumerate(
                self.s._iter_target_explanations()
            ):
                if target_idx is not None and curr_target_idx != target_idx:
                    continue
                suffix = (
                    f"target {target.value if target is not None else curr_target_idx}"
                )
                fig = self.visualize_unit_list(
                    units[curr_target_idx], draw_title=draw_title, title_suffix=suffix
                )
                figs.append(fig)
            if return_fig:
                return figs
            for fig in figs:
                fig.show()
            return

        fig = self.visualize_unit_list(units, draw_title=draw_title)
        if return_fig:
            return fig
        plt.show()

    def get_explanation_images(
        self, seed: int = 42
    ) -> dict[str, PILImage] | list[dict[str, PILImage]]:
        units = self.s.explanation_units()
        image = self.s.sample.image.content.resize((512, 512))
        if self.s._explanation_mode == "multi":
            # get explanation metadata nad target lables
            rng = np.random.RandomState(seed)
            target_units = units[rng.choice(len(units))]  # type: ignore
            return {
                unit.name: unit.draw(image)
                for unit in target_units
                # if not isinstance(unit, AggregateTextExplanationUnit)
            }
        single_units = cast(list[Any], units)
        return {
            unit.name: unit.draw(image)
            for unit in single_units
            # if not isinstance(unit, AggregateTextExplanationUnit)
        }

    @staticmethod
    def visualize_batch(
        explanation_summaries: list["ExplanationSummarizer"], seed: int = 42
    ):
        raw_grids = [
            summary.get_explanation_images() for summary in explanation_summaries
        ]
        image_grid = [entry for entry in raw_grids]

        nrow = len(image_grid)
        column_names = list(image_grid[0].keys())
        ncol = len(column_names)
        explainer_name = explanation_summaries[0].explainer
        explainer_name_map = {
            "grad/integrated_gradients": "Integrated Gradients",
            "perturbation/occlusion": "Occlusion",
            "attn/raw_attention": "Raw Attention",
        }
        explainer_name = explainer_name_map.get(explainer_name, explainer_name)

        try:
            class_labels = [
                summary.sample.get_annotation_by_type(
                    annotation_type=AnnotationType.classification
                ).label
                for summary in explanation_summaries
            ]
            predicted_class_labels = [
                summary.model_output.predicted_label_name
                for summary in explanation_summaries
            ]
            print("True class labels:", class_labels)
            print("Predicted class labels:", predicted_class_labels)
        except Exception as e:
            print("Could not extract class labels for titles:", e)
            class_labels = None

        uneven_cols = True
        for row in image_grid:
            if list(row.keys()) != column_names:
                uneven_cols = True

        fig = plt.figure(figsize=(ncol * 8, nrow * 8))
        gs = gridspec.GridSpec(
            nrow,
            ncol,
            wspace=0.0,
            hspace=0.08 if uneven_cols else 0.0,
            top=0.95,
            bottom=0.0,
            left=0.0,
            right=1.0,
        )
        for i in range(nrow):
            label = class_labels[i] if class_labels else None
            if not isinstance(label, str) and label is not None:
                label = label.name
            column_names = list(image_grid[i].keys())
            for j, col_name in enumerate(column_names):
                ax = plt.subplot(gs[i, j])
                img = image_grid[i][col_name]
                ax.imshow(img)
                ax.axis("off")

                if i == 0 or uneven_cols:
                    # if j == 0 and i == 0:
                    #     rect = patches.Rectangle(
                    #         (0, 1.05),
                    #         0.5,
                    #         0.05,
                    #         transform=ax.transAxes,
                    #         color="gray",
                    #         alpha=0.25,
                    #         clip_on=False,
                    #     )
                    #     ax.add_patch(rect)
                    #     ax.text(
                    #         0.25,
                    #         1.075,
                    #         explainer_name,
                    #         fontsize=18,
                    #         color="black",
                    #         ha="center",
                    #         va="center",
                    #         transform=ax.transAxes,
                    #         clip_on=False,
                    #     )

                    rect = patches.Rectangle(
                        (0, 1.0),
                        1.0,
                        0.05,
                        transform=ax.transAxes,
                        color="gray",
                        alpha=0.25,
                        clip_on=False,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        0.5,
                        1.025,
                        col_name + f" ({label})" if label else col_name,
                        fontsize=18,
                        color="black",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        clip_on=False,
                    )
        return fig
