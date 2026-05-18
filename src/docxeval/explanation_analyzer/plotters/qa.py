import argparse
from pathlib import Path

from docxeval.explanation_analyzer.plotters.plotter import Plotter


def process_datasets(base_dir: str) -> None:
    summarizer = Plotter(base_dir=base_dir)
    # output_dir = Path(base_dir).with_name("summarized_metrics") / "qa"
    output_dir = Path(
        "/media/saifullah/ataraxia2/phd-2026/docxeval-project/paper/mmdocxai-paper/images/results/qa"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summarizer.run(
        mode="runtime",
        output_dir=output_dir,
        datasets=[
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
