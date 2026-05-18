from typing import Any

import numpy as np
from atria_types._generic._bounding_box import BoundingBox
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage
from pydantic import BaseModel, ConfigDict, model_serializer, model_validator

from docxeval.explanation_analyzer.utils.viz import score_to_color_map


class TextExplanationUnit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: list[str]
    bboxes: list[BoundingBox]
    attribution: np.ndarray
    context_text: list[str] | None = None
    predicted_answer: str | None = None
    context_attribution: np.ndarray | None = None
    target_text: str | None = None
    target_word_bbox: BoundingBox | None = None
    target_label: str | None = None

    @property
    def name(self) -> str:
        if self.target_label is not None and self.target_text is not None:
            return f"Text ({self.target_text}, {self.target_label})"
        return "Text"

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bboxes": [bbox.value for bbox in self.bboxes],
            "attribution": self.attribution.tolist(),
            "context_text": self.context_text,
            "context_attribution": (
                self.context_attribution.tolist()
                if self.context_attribution is not None
                else None
            ),
        }

    @model_validator(mode="before")
    def validate(cls, data: dict[str, Any]) -> dict[str, Any]:
        if (
            "bboxes" in data
            and isinstance(data["bboxes"], list)
            and len(data["bboxes"]) > 0
        ):
            if isinstance(data["bboxes"][0], BoundingBox):
                return data
            elif isinstance(data["bboxes"][0], dict) and "value" in data["bboxes"][0]:
                data["bboxes"] = [
                    BoundingBox(**bbox_dict) for bbox_dict in data["bboxes"]
                ]
                return data
        else:
            return data

    @model_validator(mode="after")
    def validate_shapes(cls, instance: "TextExplanationUnit") -> "TextExplanationUnit":
        total_features = len(instance.text)
        assert (
            len(instance.bboxes) == total_features
        ), f"Number of bboxes ({len(instance.bboxes)}) must match number of text tokens ({total_features})."
        assert (
            len(instance.attribution) == total_features
        ), f"Length of attribution ({len(instance.attribution)}) must match number of text tokens ({total_features})."
        return instance

    def draw(self, image: PILImage):
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        bboxes = self.bboxes

        if self.bboxes[0].normalized:
            bboxes = [
                bbox.ops.unnormalize(width=image.width, height=image.height)
                for bbox in bboxes
            ]

        score_cmaps = score_to_color_map(self.attribution)
        for color, bbox in zip(score_cmaps, bboxes):
            x1, y1, x2, y2 = bbox.value
            draw.rectangle(
                [(x1, y1), (x2, y2)], fill=tuple(int(c * 255) for c in color)
            )

        if self.target_word_bbox is not None:
            assert (
                self.target_text is not None
            ), "target_text must be provided if target_word_bbox is provided"
            target_bbox = self.target_word_bbox.ops.unnormalize(
                width=image.width, height=image.height
            )
            draw.rectangle(
                [(target_bbox.x1, target_bbox.y1), (target_bbox.x2, target_bbox.y2)],
                outline=(0, 255, 0, 255),
            )
        composited_image = Image.alpha_composite(image.convert("RGBA"), overlay)

        if isinstance(self, TextExplanationUnit) and self.context_text is not None:
            composited_image = self.draw_context(composited_image)

        return composited_image

    def draw_context(self, image: PILImage) -> PILImage | None:
        if self.context_text is None or self.context_attribution is None:
            return None

        font_size = 12
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            font = ImageFont.load_default()

        padding = 4
        line_height = (
            font.getbbox("Ay")[3] + padding
        )  # height of text line with padding
        margin = 6

        score_cmaps = score_to_color_map(self.context_attribution)

        # --- First pass: measure banner height ---
        x, lines = margin, 1
        for token in self.context_text:
            token_display = token.replace("▁", " ").replace("Ġ", " ")
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
                (0, 0), token_display, font=font
            )
            token_width = bbox[2] - bbox[0] + padding * 2 + 2
            if x + token_width > image.width - margin:
                x = margin
                lines += 1
            x += token_width
        banner_height = lines * line_height + margin * 2

        # if answer is present, measure its width and add an extra line if it doesn't fit in the remaining space
        if self.predicted_answer is not None:
            answer_text = f"Answer: {self.predicted_answer}"
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
                (0, 0), answer_text, font=font
            )
            answer_width = bbox[2] - bbox[0] + padding * 2 + 2
            if x + answer_width > image.width - margin:
                banner_height += line_height

        # --- Second pass: draw banner ---
        banner = Image.new("RGBA", (image.width, banner_height), (245, 245, 245, 255))
        draw = ImageDraw.Draw(banner)

        x, y = margin, margin
        for token, color in zip(self.context_text, score_cmaps):
            token_display = token.replace("▁", " ").replace("Ġ", " ")
            bbox = draw.textbbox((x, y), token_display, font=font)
            token_width = bbox[2] - bbox[0] + padding * 2 + 2

            banner_rgb = (245, 245, 245)
            rgba = tuple(int(c * 255) for c in color)
            r, g, b, a = rgba
            alpha = a / 255.0

            blended = (
                int(alpha * r + (1 - alpha) * banner_rgb[0]),
                int(alpha * g + (1 - alpha) * banner_rgb[1]),
                int(alpha * b + (1 - alpha) * banner_rgb[2]),
                255,
            )
            if x + token_width > image.width - margin:
                x = margin
                y += line_height

            draw.rectangle(
                [(x, y), (x + token_width, y + line_height - 2)],
                fill=blended,
                outline=None,
            )
            draw.text((x + padding, y), token_display, fill=(0, 0, 0, 255), font=font)
            x += token_width + 2

        # draw the answer if available
        if self.predicted_answer is not None:
            answer_text = f"Answer: {self.predicted_answer}"
            bbox = draw.textbbox((x, y), answer_text, font=font)
            answer_width = bbox[2] - bbox[0] + padding * 2 + 2
            if x + answer_width > image.width - margin:
                x = margin
                y += line_height
            draw.text((x + padding, y), answer_text, fill=(0, 0, 0, 255), font=font)

        # --- Stack banner on top, original image below untouched ---
        combined = Image.new(
            "RGBA", (image.width, banner_height + image.height), (255, 255, 255, 255)
        )
        combined.paste(banner, (0, 0))
        combined.paste(image.convert("RGBA"), (0, banner_height))
        return combined


class TextPositionExplanationUnit(TextExplanationUnit):
    @property
    def name(self) -> str:
        if self.target_label is not None and self.target_text is not None:
            return f"Position ({self.target_text}, {self.target_label})"
        return "Position"


class TextLayoutExplanationUnit(TextExplanationUnit):
    @property
    def name(self) -> str:
        if self.target_label is not None and self.target_text is not None:
            return f"Layout ({self.target_text}, {self.target_label})"
        return "Layout"


class AggregateTextExplanationUnit(TextExplanationUnit):
    @property
    def name(self) -> str:
        if self.target_label is not None and self.target_text is not None:
            return f"Agg. Text ({self.target_label})"
        return "Agg. Text"

    @classmethod
    def from_text_explanation_units(cls, explanations: list[TextExplanationUnit]):
        aggregated_attribution = np.mean(
            [explanation.attribution for explanation in explanations], axis=0
        )

        return cls(
            text=explanations[0].text,
            bboxes=explanations[0].bboxes,
            attribution=aggregated_attribution,
        )


class ImageExplanationUnit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    attribution: np.ndarray
    target_text: str | None = None
    target_word_bbox: BoundingBox | None = None
    target_label: str | None = None

    @property
    def name(self) -> str:
        if self.target_label is not None and self.target_text is not None:
            return f"Image ({self.target_text}, {self.target_label})"
        return "Image"

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {"attribution": self.attribution.tolist()}

    @model_validator(mode="before")
    def validate(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "attribution" in data and isinstance(data["attribution"], list):
            data["attribution"] = np.array(data["attribution"])
        return data

    def draw(self, image):
        # create score_to_color_map(self.scores)
        colors = score_to_color_map(self.attribution)

        # clip and map to uint8
        colors = (colors * 255).clip(0, 255).astype(np.uint8)

        # make Image
        score_image = Image.fromarray(colors, mode="RGBA")

        # Resize to match input image size
        score_image = score_image.resize(image.size, Image.Resampling.NEAREST)

        # Blend with original image
        overlay = Image.alpha_composite(image.convert("RGBA"), score_image)

        if self.target_word_bbox is not None and self.target_text is not None:
            target_bbox = self.target_word_bbox.ops.unnormalize(
                width=image.width, height=image.height
            )
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(
                [(target_bbox.x1, target_bbox.y1), (target_bbox.x2, target_bbox.y2)],
                outline=(0, 255, 0, 50),
            )

        return overlay


class ImagePatchExplanationUnit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mask: np.ndarray  # [H, W] int array, patch IDs 1..N
    attribution: (
        np.ndarray
    )  # [N] attribution values; attribution[i] belongs to patch ID i+1

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            "mask": self.mask.tolist(),
            "attribution": self.attribution.tolist(),
        }

    @model_validator(mode="before")
    def validate(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "mask" in data and isinstance(data["mask"], list):
            data["mask"] = np.array(data["mask"])
        else:
            raise ValueError(
                "Invalid format for mask. Expected list that can be converted to numpy array."
            )

        if "attribution" in data and isinstance(data["attribution"], list):
            data["attribution"] = np.array(data["attribution"])
        else:
            raise ValueError(
                "Invalid format for attribution. Expected list that can be converted to numpy array."
            )

        return data

    @property
    def name(self) -> str:
        return "Image Explanation"

    def draw(self, image: PILImage):
        w, h = image.size
        mask_h, mask_w = self.mask.shape

        # nearest-neighbour resize mask to image dimensions without dtype loss
        row_idx = (np.arange(h) * mask_h / h).astype(int).clip(0, mask_h - 1)
        col_idx = (np.arange(w) * mask_w / w).astype(int).clip(0, mask_w - 1)
        mask_resized = self.mask[np.ix_(row_idx, col_idx)]

        score_cmaps = score_to_color_map(self.attribution)
        colors_uint8 = (score_cmaps * 255).clip(0, 255).astype(np.uint8)

        # build a lookup table: patch ID i (1-indexed) -> RGBA color
        max_id = int(mask_resized.max())
        lut = np.zeros((max_id + 1, 4), dtype=np.uint8)
        for i in range(len(self.attribution)):
            if i + 1 <= max_id:
                lut[i + 1] = colors_uint8[i]
        overlay_arr = lut[mask_resized]

        overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
        return Image.alpha_composite(image.convert("RGBA"), overlay_img)
