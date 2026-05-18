import json
import os
import subprocess
from pathlib import Path

import pandas as pd

RENAME_MAP = {
    "image": "Image",
    "token_ids": "Text",
    "position_ids": "Position",
    "layout_ids": "Spatial Position",
    "mean": "Mean",
    "black": "Black",
    "white": "White",
    "random": "Random",
    "mask_token_id": "[MASK] Token",
    "pad_token_id": "[PAD] Token",
    "zero": "Zero Embedding",
}

MODELS_MAP = {
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa",
    "lilt-roberta-base": "LiLT",
    "layoutlmv3-base": "LayoutLMv3",
}

DATASETS_MAP = {
    "tobacco3482_image_with_ocr": "Tobacco3482",
    "rvlcdip_image_with_ocr": "RVL-CDIP",
    "doclaynet_default": "DocLayNet",
}

data = {
    "token_ids": "zero",
    "token_type_ids": "zero",
    "position_ids": "zero",
    "layout_ids": "zero",
    "image": "black",
}


def extract_data(data: dict, target_metric: str):
    baseline_generator = data["perturbation_transform"]["baseline_generator"]
    ignored_feature_ids = data["perturbation_transform"]["ignored_feature_ids"]
    modalities_in_run = {
        key: baseline_generator[key]
        for key in [
            "token_ids",
            "token_type_ids",
            "position_ids",
            "layout_ids",
            "image",
        ]
        if key not in ignored_feature_ids
    }
    if (
        target_metric.replace("test", "PerturbationRobustnessEvaluatorStep")
        in data["metrics"]
    ):
        target_metric = target_metric.replace(
            "test", "PerturbationRobustnessEvaluatorStep"
        )
    metric_value = data["metrics"][target_metric]
    percent_features_perturbed = data["perturbation_transform"][
        "percent_features_perturbed"
    ]
    return {
        **modalities_in_run,
        "percent_features_perturbed": percent_features_perturbed,
        "metric_value": metric_value,
        "seed": data["run_idx"],
        "is_baseline": percent_features_perturbed == 0.0,
    }


def extract_and_visualize(
    eval_dir: str = "../outputs/experiment_01_seq_cls/",
    output_dir: str = "../analysis_outputs/experiment_01_seq_cls/",
    target_metric: str = "test/accuracy",
):
    # create output directory if it does not exist
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    all_extracted_data = []
    baseline_perfs = {}
    for dataset_dir in os.listdir(eval_dir):
        for model_dir in os.listdir(os.path.join(eval_dir, dataset_dir)):
            data_dir = os.path.join(eval_dir, dataset_dir, model_dir)
            dataset = dataset_dir.replace("/", "_")
            metrics_file = os.path.join(data_dir, "metrics.json")
            with open(metrics_file, "r") as f:
                run_data = json.load(f)
                for hash, d in run_data.items():
                    extracted_data = extract_data(d, target_metric=target_metric)
                    if extracted_data["is_baseline"]:
                        baseline_perfs[f"{dataset}_{model_dir}"] = extracted_data[
                            "metric_value"
                        ]
                        continue

                    extracted_data["dataset"] = dataset
                    extracted_data["model"] = model_dir
                    all_extracted_data.append(extracted_data)

    df = pd.DataFrame(all_extracted_data)

    # lets add baseline performance for each model-dataset combination to the dataframe for easy comparison
    df["baseline_perf"] = df.apply(
        lambda row: baseline_perfs.get(f"{row['dataset']}_{row['model']}", None), axis=1
    )

    # filter out rows
    df = df[df["image"] != "black"]
    df = df[df["image"] != "white"]

    # first group by model and dataset, then group over a specific unique value of each modality with in each group
    grouped = df.groupby(["model", "dataset"])
    modality_columns = [
        "token_ids",
        "position_ids",
        "layout_ids",
        "image",
    ]

    # aggregated_data
    agg_df = []
    for (model, dataset), group in grouped:
        for modality in modality_columns:
            assert modality in group.columns, f"Modality {modality} not in columns"
            modality_grouped = group.groupby(
                [modality, "seed", "percent_features_perturbed"]
            )
            for (strategy, seed, percent_perturbed), subgroup in modality_grouped:
                if strategy == "none":
                    continue

                agg_df.append(
                    {
                        "model": MODELS_MAP.get(model, model),
                        "dataset": DATASETS_MAP.get(dataset, dataset),
                        "modality_strategy_pair": f"{modality}={strategy}",
                        "modality": RENAME_MAP[modality],
                        "strategy_type": RENAME_MAP[strategy],
                        "seed": seed,
                        "percent_features_perturbed": percent_perturbed,
                        "perf": subgroup["metric_value"].mean(),
                        "baseline_perf": subgroup["baseline_perf"].mean(),
                    }
                )

    # Write all modality rankings to a single file
    ranked_strategies_file = Path(output_dir) / "ranked_strategies_all_modalities.txt"
    with open(ranked_strategies_file, "w") as f:
        f.write("dataset,model,modality,ranked_strategies\n")
    agg_df = pd.DataFrame(agg_df)

    # remove baseline perf to get drop in perf
    agg_df["perf"] = agg_df["perf"] - agg_df["baseline_perf"]

    # Open file once for all model-dataset combinations
    for (model, dataset), group in agg_df.groupby(["model", "dataset"]):
        save_dir = Path(output_dir) / dataset / model
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        file_path = save_dir / "pert_robustness_modality_wise_mean_perf.csv"
        group.to_csv(file_path, index=False)

        # for each modality, rank strategies and write to file
        for modality in group["modality"].unique():
            modality_group = group[group["modality"] == modality]
            # Group by strategy_type and percent_features_perturbed, calculate mean perf
            strategy_perf = (
                modality_group.groupby(["strategy_type", "percent_features_perturbed"])[
                    "perf"
                ]
                .mean()
                .reset_index()
            )

            # Rank strategies within each percent_features_perturbed group
            strategy_perf["rank"] = strategy_perf.groupby("percent_features_perturbed")[
                "perf"
            ].rank(ascending=False, method="average")

            # Calculate mean rank across all percent_features_perturbed levels
            ranked_strategies = (
                strategy_perf.groupby("strategy_type")["rank"]
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )

            # Merge with mean performance for display
            ranked_strategies = ranked_strategies.merge(
                strategy_perf.groupby("strategy_type")["perf"].mean().reset_index(),
                on="strategy_type",
            )
            strategies_str = ", ".join(
                f"{row['strategy_type']} ({idx + 1})"
                for idx, row in ranked_strategies.iterrows()
            )

            with open(ranked_strategies_file, "a") as f:
                f.write(f"{dataset},{model},{modality},{strategies_str}\n")

            print("Visualizing results for", dataset, model)
            visualize_results(file_path, dataset)


def visualize_results(file_path, dataset):
    try:
        output_path = file_path.parent / "box_plot_pert_robustness_comparison.png"
        BASE_PATH = "src/docxeval/analysis/robustness/R"
        script_path = os.path.abspath(
            f"{BASE_PATH}/box_plot_pert_robustness_comparison.R"
        )
        assert Path(script_path).exists(), "R script not found!"
        result = subprocess.run(
            [
                "Rscript",
                script_path,
                file_path,
                output_path,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        exit()


if __name__ == "__main__":
    for config in zip(
        [
            "experiment_01_seq_cls",
            "experiment_01_token_cls",
            # "experiment_01_layout_token_cls",
            "experiment_01_qa_cls",
        ],
        [
            "test/accuracy",
            "test/seqeval/f1_score",
            # "test/layout_f1_macro",
            "test/due_eval/ANLS",
        ],
    ):
        dir, metric = config
        extract_and_visualize(
            eval_dir=f"../outputs/{dir}/",
            output_dir=f"../analysis_outputs/{dir}/",
            target_metric=metric,
        )
