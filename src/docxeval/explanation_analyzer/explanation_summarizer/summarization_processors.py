from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExplanationSummaryProcessor(ABC):
    """Base interface for per-summary explanation analysis processors."""

    name: str = "base"

    def __init__(self) -> None:
        self._run_dir: Path | None = None
        self._output_dir: Path | None = None
        self._config: Any = None

    def initialize(self, run_dir: Path, config: Any) -> None:
        """Prepare processor state and output directory before processing."""
        self._run_dir = Path(run_dir)
        self._config = config
        self._output_dir = self._run_dir / self.name
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        assert self._output_dir is not None, "Processor is not initialized."
        return self._output_dir

    def is_done(self) -> bool:
        """Return True if this processor has already been finalized and can be skipped."""
        return False

    @abstractmethod
    def process_summary(self, summary: Any) -> None:
        """Consume one explanation summary and store intermediate rows/state."""

    @abstractmethod
    def finalize(self) -> dict[str, str | int | float]:
        """Write final data artifacts and return artifact metadata."""

    def finalize_plots(self) -> None:
        """Re-run plotting on already-finalized data. No-op by default."""
        pass


def get_registered_expl_analysis_processors() -> list[ExplanationSummaryProcessor]:
    """Return all processors to run for mode='expl_analysis'."""
    from docxeval.explanation_analyzer.utils.modality_contribution import (
        ModalityContributionProcessor,
    )

    return [
        ModalityContributionProcessor(),
    ]
