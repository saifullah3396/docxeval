import argparse
from pathlib import Path

from docxeval.explanation_analyzer.plotters.plotter import Plotter


def process_datasets(base_dir: str) -> None:
    summarizer = Plotter(base_dir=base_dir)
    output_dir = Path(
        "/media/saifullah/ataraxia2/phd-2026/docxeval-project/paper/mmdocxai-paper/images/results/all_datasets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in [
        "metric_table",
        "critical_difference",
        "rankings",
        "correlations",
        "modality_frac_comp_table",
    ]:
        summarizer.run(
            mode=mode,
            output_dir=output_dir,
            datasets=[
                "tobacco3482_image_with_ocr",
                "rvlcdip_image_with_ocr",
                "doclaynet_default",
                "cord",
                "funsd",
                "sroie",
                "wild_receipts",
                "due_benchmark_DocVQA",
            ],
        )


if __name__ == "__main__":
    # make a arg parser with base dir as input
    parser = argparse.ArgumentParser(description="Process datasets for CLS sequence.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing the explanation analysis outputs.",
    )
    args = parser.parse_args()
    process_datasets(base_dir=args.base_dir)
