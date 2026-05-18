import pandas as pd


def compute_task_wise_modality_ranking(
    eval_dir: str,
):
    # get the dataset/model wise ranking file
    dataset_model_wise_ranking = f"{eval_dir}/ranked_strategies_all_modalities.txt"

    data = []
    with open(dataset_model_wise_ranking, "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if i == 0:
                continue
            parts = line.strip().split(",")
            dataset = parts[0]
            model = parts[1]
            modality = parts[2]
            rankings = parts[3:]

            # sanitize rankings
            for rank, ranking in enumerate(rankings):
                data.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "modality": modality,
                        "strategy_type": ranking.split("(")[0].strip(),
                        "rank": rank + 1,
                    }
                )
    df = pd.DataFrame(data)
    for (model, modality), group in df.groupby(["model", "modality"]):
        group = group.reset_index(drop=True)
        ranked = (
            group.groupby("strategy_type")["rank"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        print("Model:", model, " Modality:", modality)
        for idx, row in ranked.iterrows():
            print(f"{idx + 1}. {row['strategy_type']} (Mean Rank: {row['rank']:.2f})")


if __name__ == "__main__":
    for dir in [
        # "experiment_01_seq_cls",
        # "experiment_01_token_cls",
        # "experiment_01_layout_token_cls",
        "experiment_01_qa_cls",
    ]:
        compute_task_wise_modality_ranking(
            eval_dir=f"../analysis_outputs/{dir}/",
        )
