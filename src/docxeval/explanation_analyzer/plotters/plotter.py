import glob
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from critdd import tikz_2d
from critdd.diagram import Diagram, Diagrams

from docxeval.explanation_analyzer.plotters._helpers import (
    collapse_labels,
    draw_locality_confusion,
    extract_diagnostic_scalars,
    select_top_labels,
)


class MyDiagrams(Diagrams):
    def __init__(
        self,
        Xs,
        *,
        diagram_names=None,
        treatment_names=None,
        maximize_outcome=False,
    ):
        n_diagrams = len(Xs)
        n_treatments = Xs[0].shape[1]
        if not np.all([X.shape[1] == n_treatments for X in Xs]):
            raise ValueError("Xs has elements with different numbers of treatments")
        if diagram_names is None:
            diagram_names = [f"diagram {i + 1}" for i in range(n_diagrams)]
        elif len(diagram_names) != n_diagrams:
            raise ValueError("len(diagram_names) != len(Xs)")
        if treatment_names is None:
            treatment_names = [f"treatment {i + 1}" for i in range(n_treatments)]
        elif len(treatment_names) != n_treatments:
            raise ValueError("len(treatment_names) != Xs[i].shape[1]")
        self.diagram_names = diagram_names
        self.diagrams = [
            Diagram(
                X, treatment_names=treatment_names, maximize_outcome=maximize_outcome
            )
            for X in Xs
        ]
        self.n_observations = Xs[0].shape[0]
        print("self.n_observations", self.n_observations)

    def to_str(self, alpha=0.05, adjustment="holm", **kwargs):
        # lets update it so that if we have very few samples we ignore the groups and just plot the average ranks without any grouping lines. We can do this by checking the number of samples in each diagram and if it's below a certain threshold (e.g., 10), we skip the grouping step.

        groups = []
        for d in self.diagrams:
            if self.n_observations < 11:
                groups.append([])  # No groups, just plot average ranks
            else:
                groups.append(d.get_groups(alpha, adjustment, return_singletons=False))
        return tikz_2d.to_str(
            np.stack([d.average_ranks for d in self.diagrams]),
            groups,
            self.treatment_names,
            self.diagram_names,
            **kwargs,
        )


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("[%(levelname)s] %(message)s"))

logger.handlers.clear()
logger.addHandler(handler)


# Color and marker definitions for up to 20 treatments
colors = [
    "84B818",
    "D18B12",
    "1BB5B5",
    "F85A3E",
    "4B6CFC",
    "E377C2",
    "7F7F7F",
    "BCBD22",
    "17BECF",
    "AEC7E8",
    "FFBB78",
    "FF7F0E",
    "2CA02C",
    "D62728",
    "9467BD",
    "8C564B",
    "E377C2",
    "7F7F7F",
    "BCBD22",
    "17BECF",
]
markers = [
    "*",
    "diamond*",
    "triangle,semithick",
    "square,semithick",
    "pentagon,semithick",
    "star,semithick",
    "otimes*,semithick",
    "+,semithick",
    "x,semithick",
    # "|,semithick",
    "asterisk,semithick",
    "Mercedes star,semithick",
    "oplus*,semithick",
    "diamond,semithick",
    "triangle*,semithick",
    "square*,semithick",
    "star*,semithick",
    "otimes,semithick",
    "+,semithick",
    "x,semithick",
]


@dataclass
class DisplayConfig:
    """Human-readable ordering and renaming for display."""

    model_order: list[str] = field(
        default_factory=lambda: [
            "bert-base-uncased",
            "roberta-base",
            "lilt-roberta-base",
            "layoutlmv3-base",
        ]
    )
    explainer_order: list[str] = field(
        default_factory=lambda: [
            "saliency",
            "input_x_gradient",
            "guided_backprop",
            "deeplift",
            "integrated_gradients",
            "deeplift_shap",
            "gradient_shap",
            "feature_ablation",
            "occlusion",
            "lime",
            "kernel_shap",
            "raw_attention",
            "attention_rollout",
        ]
    )
    dataset_order: list[str] = field(
        default_factory=lambda: [
            "tobacco3482_image_with_ocr",
            "rvlcdip_image_with_ocr",
            "doclaynet_default",
            "funsd",
            "cord",
            "sroie",
            "wild_receipts",
            "due_benchmark_DocVQA",
        ]
    )
    model_rename: dict[str, str] = field(
        default_factory=lambda: {
            "bert-base-uncased": "BERT",
            "roberta-base": "RoBERTa",
            "lilt-roberta-base": "LiLT",
            "layoutlmv3-base": "LayoutLMv3",
        }
    )
    dataset_rename: dict[str, str] = field(
        default_factory=lambda: {
            "tobacco3482_image_with_ocr": "Tobacco3482",
            "rvlcdip_image_with_ocr": "RVL-CDIP",
            "doclaynet_default": "DocLayNet",
            "funsd": "FUNSD",
            "cord": "CORD",
            "sroie": "SROIE",
            "wild_receipts": "Wild Receipts",
            "due_benchmark_DocVQA": "DocVQA",
        }
    )
    explainer_rename: dict[str, str] = field(
        default_factory=lambda: {
            "saliency": "Sal.",
            "input_x_gradient": "IxG",
            "guided_backprop": "GB",
            "deeplift": "DL",
            "integrated_gradients": "IG",
            "deeplift_shap": "DL SHAP",
            "gradient_shap": "GS",
            "feature_ablation": "FA",
            "occlusion": "Occ.",
            "lime": "LIME",
            "kernel_shap": "K-SHAP",
            "raw_attention": "RA",
            "attention_rollout": "AR",
        }
    )
    metric_rename: dict[str, str] = field(
        default_factory=lambda: {
            "ABPC": "ABPC",
            "Sufficiency": "Suff.",
            "Sensitivity-N (60)": "Sens-N (60)",
            "Sensitivity-N (40)": "Sens-N (40)",
            "Sensitivity-N (20)": "Sens-N (20)",
            "Non-Sensitivity": "Non-Sens.",
            "Monotonicity": "Mon.",
            "Monotonicity Correlation": "Mon. Corr.",
            "Faithfulness Correlation (60)": "Faith. Corr. (60)",
            "Faithfulness Correlation (40)": "Faith. Corr. (40)",
            "Faithfulness Correlation (20)": "Faith. Corr. (20)",
            "Faithfulness Estimate": "Faith. Est.",
            "Infidelity": "Infid.",
            "Effective Complexity": "Eff. Comp.",
            "Entropy-Based Complexity": "Ent. Comp.",
            "Sparseness": "SP",
            "Sundararajan Complexity": "S. Comp.",
        }
    )


class Plotter:
    """Post-processes and summarizes explanation metrics across datasets, models,
    and explainers.

    Operates purely on the serialized output directory structure — no dependency
    on ModelExplanationAnalyzer or any model/data pipeline.

    Expected directory layout::

        base_dir/
        ├── <dataset>/
        │   └── <model>/
        │       └── <run_subdir>/
        │           └── <explainer>/
        │               └── dataset_level_metrics.json
    """

    SUMMARY_FILENAME = "dataset_level_metrics.json"

    def __init__(
        self,
        base_dir: str | Path,
        exclude_metrics: list[str] | None = None,
        display_config: DisplayConfig | None = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._exclude_metrics = exclude_metrics or []
        self._display_config = display_config or DisplayConfig()

        if not self._base_dir.exists():
            raise FileNotFoundError(f"Base directory does not exist: {self._base_dir}")

        self._df: pd.DataFrame | None = None
        self._label_wise_metrics_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def _discover_summary_files(self) -> list[Path]:
        pattern = str(self._base_dir / "**" / self.SUMMARY_FILENAME)
        files = [Path(f) for f in glob.glob(pattern, recursive=True)]
        if not files:
            logger.warning(
                f"No {self.SUMMARY_FILENAME} files found under {self._base_dir}. "
                "Run the analyzer in 'prepare_metrics' mode first."
            )
        else:
            logger.info(f"Discovered {len(files)} summary files under {self._base_dir}")
        return files

    def _build_df(self, label_wise: bool) -> pd.DataFrame:
        """Glob all summary JSONs and build a flat DataFrame.

        When ``label_wise`` is False, reads top-level ``metrics``.
        When True, reads per-label metrics from ``label_metrics`` and adds a
        ``label`` column.
        """
        files = self._discover_summary_files()
        if not files:
            return pd.DataFrame()

        source_key = "label_metrics" if label_wise else "metrics"
        records: list[dict] = []

        for filepath in files:
            explainer_name = filepath.parent.name
            with open(filepath, "r") as f:
                data = json.load(f)

            dataset_name = data.get("dataset_name")
            model_name = data.get("model_name")

            # Normalize both cases into a common iterable of
            # (label_or_none, metric_name, metric_value) triples.
            if label_wise:
                triples = (
                    (label, mname, mvalue)
                    for label, label_data in data.get(source_key, {}).items()
                    for mname, mvalue in label_data.items()
                )
            else:
                triples = (
                    (None, mname, mvalue)
                    for mname, mvalue in data.get(source_key, {}).items()
                )

            for label, metric_name, metric_value in triples:
                if metric_name in self._exclude_metrics:
                    continue

                if metric_name == "Non-Sensitivity":
                    metric_value["type"] = "axiomatic"

                score = metric_value.get("score")
                if metric_name == "AOPC":
                    score = {
                        "aopc.desc": metric_value.get("aopc.desc"),
                        "aopc.asc": metric_value.get("aopc.asc"),
                        "aopc.random": metric_value.get("aopc.rand"),
                    }
                if score is None:
                    score = metric_value

                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "explainer": explainer_name,
                    "metric_name": metric_name,
                    "metric_value": score,
                    "exec_time": metric_value.get("exec_time"),
                    "metric_type": metric_value["type"],
                    "is_lower_the_better": metric_value.get("is_lower_the_better"),
                    "metric_perturbation_type": metric_value.get(
                        "metric_perturbation_type"
                    ),
                }
                if label_wise:
                    record["label"] = label
                records.append(record)

                if metric_name == "AOPC":
                    score_desc = metric_value.get("aopc.desc")
                    score_rand = metric_value.get("aopc.rand")
                    if (
                        isinstance(score_desc, list)
                        and isinstance(score_rand, list)
                        and len(score_desc) > 0
                        and len(score_rand) > 0
                    ):
                        aopc_score = score_desc[-1] - score_rand[-1]
                    else:
                        aopc_score = np.nan
                    record = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "explainer": explainer_name,
                        "metric_name": "AOPC (desc.)",
                        "metric_value": aopc_score,
                        "exec_time": metric_value.get("exec_time"),
                        "metric_type": metric_value["type"],
                        "is_lower_the_better": metric_value.get("is_lower_the_better"),
                        "metric_perturbation_type": metric_value.get(
                            "metric_perturbation_type"
                        ),
                    }
                    if label_wise:
                        record["label"] = label
                    records.append(record)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df

        # Warn on NaN (ignoring None).
        nan_mask = df.map(lambda x: isinstance(x, float) and np.isnan(x))
        if nan_mask.any().any():
            nan_rows = df[nan_mask.any(axis=1)]
            logger.warning(
                f"Found {len(nan_rows)} records with NaN values. "
                "This may indicate missing metrics or incomplete runs."
            )
            logger.warning(f"Example NaN records:\n{nan_rows.head()}")

        logger.info(
            f"Aggregated {len(df)} records "
            f"({'label-wise' if label_wise else 'overall'}): "
            f"{df['dataset'].nunique()} dataset(s), "
            f"{df['model'].nunique()} model(s), "
            f"{df['explainer'].nunique()} explainer(s)"
        )

        # Normalize so higher is always better.
        df["metric_value"] = df.apply(
            lambda row: (
                -row["metric_value"]
                if row["is_lower_the_better"] and row["metric_name"] != "AOPC"
                else row["metric_value"]
            ),
            axis=1,
        )
        return df

    @staticmethod
    def _filter(
        df: pd.DataFrame,
        datasets: list[str] | None,
        models: list[str] | None,
        explainers: list[str] | None,
    ) -> pd.DataFrame:
        if datasets:
            df = df[df["dataset"].isin(datasets)]
        if models:
            df = df[df["model"].isin(models)]
        if explainers:
            df = df[df["explainer"].isin(explainers)]
        return df

    def aggregate(
        self,
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        explainers: list[str] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Flat DataFrame of overall (non-label-wise) metrics. Cached."""
        if self._df is None or force:
            self._df = self._build_df(label_wise=False)
        df = self._filter(self._df, datasets, models, explainers)
        if not df.empty:
            logger.info(
                f"Current aggregated {len(df)} records: "
                f"{df['dataset'].nunique()} dataset(s), "
                f"{df['model'].nunique()} model(s), "
                f"{df['explainer'].nunique()} explainer(s)"
            )
        return df

    def aggregate_label_wise(
        self,
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        explainers: list[str] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Flat DataFrame of label-wise metrics (adds ``label`` column). Cached."""
        if self._label_wise_metrics_df is None or force:
            self._label_wise_metrics_df = self._build_df(label_wise=True)
        df = self._filter(self._label_wise_metrics_df, datasets, models, explainers)
        if not df.empty:
            logger.info(
                f"Current aggregated {len(df)} records: "
                f"{df['dataset'].nunique()} dataset(s), "
                f"{df['model'].nunique()} model(s), "
                f"{df['explainer'].nunique()} explainer(s)"
            )
        return df

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def with_display_names(
        self,
        df: pd.DataFrame,
        model_order: list[str] | None = None,
        explainer_order: list[str] | None = None,
        dataset_order: list[str] | None = None,
        model_rename: dict[str, str] | None = None,
        dataset_rename: dict[str, str] | None = None,
        metric_rename: dict[str, str] | None = None,
        explainer_rename: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Apply categorical ordering and human-readable renaming.

        Per-call arguments override the instance DisplayConfig.
        Returns a copy.
        """
        cfg = self._display_config
        model_order = model_order or cfg.model_order
        explainer_order = explainer_order or cfg.explainer_order
        dataset_order = dataset_order or cfg.dataset_order
        model_rename = model_rename or cfg.model_rename
        dataset_rename = dataset_rename or cfg.dataset_rename
        metric_rename = metric_rename or cfg.metric_rename
        explainer_rename = explainer_rename or cfg.explainer_rename

        df = df.copy()
        if model_order:
            existing_models = list(pd.unique(df["model"].dropna()))
            ordered_models = [m for m in model_order if m in existing_models] + [
                m for m in existing_models if m not in model_order
            ]
            df["model"] = pd.Categorical(
                df["model"], categories=ordered_models, ordered=True
            )

        if dataset_order:
            existing_datasets = list(pd.unique(df["dataset"].dropna()))
            ordered_datasets = [d for d in dataset_order if d in existing_datasets] + [
                d for d in existing_datasets if d not in dataset_order
            ]
            df["dataset"] = pd.Categorical(
                df["dataset"], categories=ordered_datasets, ordered=True
            )

        if explainer_order:
            existing_explainers = list(pd.unique(df["explainer"].dropna()))
            ordered_explainers = [
                e for e in explainer_order if e in existing_explainers
            ] + [e for e in existing_explainers if e not in explainer_order]
            df["explainer"] = pd.Categorical(
                df["explainer"], categories=ordered_explainers, ordered=True
            )

        df.sort_values(["dataset", "model", "explainer"], inplace=True)

        if model_rename:
            df["model"] = df["model"].cat.rename_categories(model_rename)
        if dataset_rename:
            df["dataset"] = df["dataset"].cat.rename_categories(dataset_rename)
        if metric_rename:
            df["metric_name"] = df["metric_name"].replace(metric_rename)
        if explainer_rename:
            df["explainer"] = df["explainer"].cat.rename_categories(explainer_rename)

        return df

    # ------------------------------------------------------------------
    # Analysis: metric table
    # ------------------------------------------------------------------

    def compute_metric_table(
        self,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Pivot all metrics into one wide table indexed by (dataset, model, explainer).

        Columns are ``{metric_type}_{metric_name}``.

        Returns
        -------
        pd.DataFrame
            Wide-format table with one column per metric_type + metric_name.
        """
        if df is None:
            df = self.aggregate()
        if df.empty:
            return pd.DataFrame()

        table_df = df.copy()

        # remove metrics with diagnostics type
        table_df = table_df[table_df["metric_type"] != "diagnostics"]

        # make sure to only take values with floats
        table_df = table_df[
            table_df["metric_value"].apply(lambda x: not isinstance(x, dict))
        ]

        table_df["metric_type_name"] = (
            table_df["metric_type"] + "_" + table_df["metric_name"]
        )
        pivot = table_df.pivot_table(
            index=["dataset", "model", "explainer"],
            columns="metric_type_name",
            values="metric_value",
        )

        logger.info(
            f"Computed metric table: {pivot.shape[0]} rows x {pivot.shape[1]} metrics"
        )
        return pivot

    def _prepare_modality_fraction_data(
        self,
        df: pd.DataFrame,
        label_wise_df: pd.DataFrame,
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        explainers: list[str] | None = None,
        top_k: int = 6,
        metric_name: str = "ModalityTopkFraction",
        seed: int = 0,
    ) -> pd.DataFrame | None:
        """Filter, reshape, and combine overall + label-wise modality fractions.

        Returns a long-form DataFrame with columns:
            dataset, explainer, label, modality, score

        ``label`` is ``"Overall"`` for dataset-level rows.  For label-level rows
        a random sample of *top_k* labels per dataset is kept.

        Returns ``None`` when no data matches *metric_name*.
        """
        import numpy as np

        # ---- 0. Apply filters ---------------------------------------------------
        def _apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            if datasets:
                frame = frame[frame["dataset"].isin(datasets)]
            if models:
                frame = frame[frame["model"].isin(models)]
            if explainers:
                frame = frame[frame["explainer"].isin(explainers)]
            return frame

        df = _apply_filters(df)
        label_wise_df = _apply_filters(label_wise_df)

        # ---- 1. Explode dict/Series metric_value into long form -----------------
        def _explode_scores(frame: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
            scores = frame["metric_value"].apply(pd.Series)
            keep = [c for c in scores.columns if c.startswith("score_")]
            wide = pd.concat(
                [
                    frame[id_cols].reset_index(drop=True),
                    scores[keep].reset_index(drop=True),
                ],
                axis=1,
            )
            long = wide.melt(
                id_vars=id_cols,
                value_vars=keep,
                var_name="modality",
                value_name="score",
            ).dropna(subset=["score"])
            long["modality"] = (
                long["modality"]
                .str.replace("score_", "", regex=False)
                .str.replace("_embeddings", "", regex=False)
            )
            return long

        # ---- 2. Reshape overall -------------------------------------------------
        overall = df[df["metric_name"] == metric_name].copy()
        if overall.empty:
            return None

        overall_long = _explode_scores(overall, ["dataset", "model", "explainer"])
        overall_long["label"] = "Overall"

        # ---- 3. Reshape label-wise & sample top-K randomly ----------------------
        label_wise = label_wise_df[label_wise_df["metric_name"] == metric_name].copy()

        label_long = pd.DataFrame()
        if not label_wise.empty:
            label_long = _explode_scores(
                label_wise, ["dataset", "model", "explainer", "label"]
            )

            label_long["label"] = (
                label_long["label"]
                .str.replace("B-", "", regex=False)
                .str.replace("I-", "", regex=False)
            )

            # group and average scores
            label_long = (
                label_long.groupby(
                    ["dataset", "model", "explainer", "label", "modality"],
                )["score"]
                .mean()
                .dropna()
                .reset_index()
            )

            # Pick top_k random labels per dataset
            rng = np.random.default_rng(seed)
            sampled_parts = []
            for ds, grp in label_long.groupby("dataset"):
                unique_labels = grp["label"].unique()
                k = min(top_k, len(unique_labels))
                chosen = rng.choice(unique_labels, size=k, replace=False)
                sampled_parts.append(grp[grp["label"].isin(chosen)])
            label_long = pd.concat(sampled_parts, ignore_index=True)

        # ---- 4. Combine ---------------------------------------------------------
        return pd.concat([overall_long, label_long], ignore_index=True)

    def plot_modality_fraction_bar(
        self,
        df: pd.DataFrame,
        label_wise_df: pd.DataFrame,
        output_path: str | Path = "modality_fraction_bar.png",
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        explainers: list[str] | None = None,
        top_k: int = 4,
        metric_name: str = "ModalityTopkFraction",
        seed: int = 42,
    ) -> None:
        """Faceted heatmap of modality contribution.

        Produces **one figure per dataset** using ``patchwork`` to stitch
        together one small heatmap per label (Overall + top-K random labels).
        Each heatmap has modalities on x, explainers on y, fill = score.

        Files are saved as ``<stem>_<dataset_name>.png``.
        """
        from pathlib import Path

        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        output_path = Path(output_path)
        combined = self._prepare_modality_fraction_data(
            df=df,
            label_wise_df=label_wise_df,
            datasets=datasets,
            models=models,
            explainers=explainers,
            top_k=top_k,
            metric_name=metric_name,
            seed=10,
        )
        if combined is None:
            logger.warning(f"No rows with metric_name='{metric_name}' in df.")
            return

        # ---- 5. One figure per (dataset, model) --------------------------------
        combos = (
            combined[["dataset", "model"]]
            .drop_duplicates()
            .sort_values(["dataset", "model"])
            .values.tolist()
        )
        saved = []

        for ds, mdl in combos:
            sub = combined[
                (combined["dataset"] == ds) & (combined["model"] == mdl)
            ].copy()

            all_labels = sorted(sub["label"].unique())
            if "Overall" in all_labels:
                all_labels.remove("Overall")
            all_labels = ["Overall"] + all_labels
            n_labels = len(all_labels)

            # Ensure every (label, explainer, modality) combo exists; missing → NaN
            import itertools

            all_explainers = sorted(sub["explainer"].unique())
            all_modalities = sorted(sub["modality"].unique())
            full_grid = pd.DataFrame(
                list(itertools.product(all_labels, all_explainers, all_modalities)),
                columns=["label", "explainer", "modality"],
            )
            sub = full_grid.merge(
                sub, on=["label", "explainer", "modality"], how="left"
            )

            ds_out = output_path.with_name(
                f"{output_path.stem}_{ds}_{mdl}{output_path.suffix}"
            )

            plot_defs = []
            for j, lbl in enumerate(all_labels):
                plot_defs.append(f"""
                p{j} <- build_heatmap(
                    plot_df[plot_df$label == "{lbl}", ],
                    "{lbl}"
                )
                """)

            patch_expr = " + ".join(f"p{j}" for j in range(n_labels))

            r_script = f"""
                library(ggplot2)
                library(patchwork)
                library(viridis)

                modality_levels <- c("token", "position", "layout", "image")
                plot_df$modality <- factor(plot_df$modality, levels = modality_levels)

                build_heatmap <- function(df, title_text) {{
                    df$text_label <- ifelse(is.na(df$score),
                                            "N/A",
                                            sprintf("%.2f", df$score))
                    df$text_color <- ifelse(is.na(df$score), "na",
                                     ifelse(df$score > 0.5, "high", "low"))

                    ggplot(df, aes(x = modality, y = explainer)) +
                        geom_tile(aes(fill = score),
                                  color = "grey65", linewidth = 0.4) +
                        geom_text(aes(label = text_label, color = text_color),
                                  size = 2.5, show.legend = FALSE) +
                        scale_color_manual(values = c("high" = "white",
                                                      "low"  = "black",
                                                      "na"   = "grey40")) +
                        scale_fill_viridis(discrete = FALSE, direction = -1,
                                           limits = c(0, 1), option = "magma",
                                           na.value = "grey90") +
                        coord_equal() +
                        labs(title = title_text) +
                        theme_minimal(base_size = 11) +
                        theme(
                            axis.title = element_blank(),
                            axis.text.x = element_text(angle = 45, hjust = 1,
                                                       vjust = 1, size = 8),
                            axis.text.y = element_text(size = 8),
                            plot.title = element_text(face = "bold",
                                                      hjust = 0.5, size = 10),
                            legend.position = "right"
                        )
                }}

                {"".join(plot_defs)}

                p <- {patch_expr} +
                    plot_layout(ncol = {n_labels}, guides = "collect") +
                    plot_annotation(
                        title = "{ds} / {mdl}",
                        theme = theme(plot.title = element_text(
                            face = "bold", size = 13, hjust = 0))
                    ) &
                    theme(legend.position = "right")

                fig_w <- max(5, {n_labels} * 3 + 1.5)
                fig_h <- 3

                ggsave(
                    out_path,
                    plot = p,
                    width = fig_w,
                    height = fig_h,
                    dpi = 300
                )
            """

            with localconverter(robjects.default_converter + pandas2ri.converter):
                robjects.globalenv["plot_df"] = robjects.conversion.py2rpy(sub)
            robjects.globalenv["out_path"] = str(ds_out)
            robjects.r(r_script)
            saved.append(ds_out)

        logger.info(f"Saved modality fraction plots: {saved}")
        return saved

    # ------------------------------------------------------------------
    # Analysis: runtime
    # ------------------------------------------------------------------

    def compute_runtime_summary(
        self,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute mean execution time per metric across all runs.

        Returns
        -------
        pd.DataFrame
            One row per metric with mean exec_time, metric_type,
            and metric_perturbation_type.
        """
        if df is None:
            df = self.aggregate()
        if df.empty:
            return pd.DataFrame()

        runtime_df = df.copy()

        # remove metrics with diagnostics type
        runtime_df = runtime_df[runtime_df["metric_type"] != "diagnostics"]

        runtime_df["metric_perturbation_type"] = runtime_df[
            "metric_perturbation_type"
        ].fillna("N/A")

        runtime_df = (
            runtime_df.groupby(
                ["metric_name", "metric_type", "metric_perturbation_type"],
                as_index=False,
            )["exec_time"]
            .mean()
            .sort_values("exec_time", ascending=True)
        )

        logger.info(f"Computed runtime summary for {len(runtime_df)} metrics")
        return runtime_df

    def draw_runtime_bar_plot(
        self,
        runtime_df: pd.DataFrame,
        output_path: str | Path = "runtime_summary_plot.png",
    ):
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        # Convert pandas → R dataframe using new API
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.conversion.py2rpy(runtime_df)

        robjects.globalenv["data"] = r_df
        robjects.globalenv["output_path"] = str(output_path)

        robjects.r("""
            library(ggplot2)
            library(scales)

            # Map perturbation_type to marker shapes
            shape_map <- c(
                "ordered_perturbation" = 22,
                "unordered_perturbation" = 21,
                "N/A" = 23
            )

            # Keep everything in seconds (no ms conversion)
            data$exec_time_sec <- data$exec_time * 1000

            # Sort bars by runtime
            data$metric_name <- factor(
                data$metric_name,
                levels = data$metric_name[order(data$exec_time_sec)]
            )

            # Plot
            p <- ggplot(data, aes(
                y = metric_name,
                x = exec_time_sec,
                fill = metric_type
            )) +
                geom_col(width = 0.6, color = "black") +
                geom_point(aes(shape = metric_perturbation_type),
                        color = "black", size = 3) +
                scale_shape_manual(values = shape_map) +
                scale_fill_brewer(palette = "Set2") +
                scale_x_continuous(
                    trans = pseudo_log_trans(base = 10),
                    breaks = c(0, 10, 100, 500, 1000, 5000, 10000, 20000)
                ) +
                theme_gray(base_size = 14) +
                theme(
                    axis.title.y = element_blank(),
                    axis.text.x  = element_text(size = 10),   # tick label font size
                    legend.position = "right"
                ) +
                labs(
                    x = "Mean Execution Time (ms)",
                    fill = "Metric Type",
                    shape = "Perturbation Type",

                ) +
                guides(
                    shape = guide_legend(override.aes = list(fill = NA, linetype = 0)),
                    fill  = guide_legend(override.aes = list(shape = NA))
                )

            # Save figure
            ggsave(output_path, plot = p, width = 10, height = 5, dpi = 300)
        """)

    # ------------------------------------------------------------------
    # Analysis: correlations
    # ------------------------------------------------------------------

    def compute_metric_correlations(
        self,
        df: pd.DataFrame | None = None,
        metric_types: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute per-metric-type correlation matrices.

        Returns
        -------
        dict[str, pd.DataFrame]
            metric_type → correlation matrix.
        """
        if df is None:
            df = self.aggregate()
        if df.empty:
            return {}

        if metric_types is None:
            metric_types = ["complexity", "faithfulness"]

        correlations: dict[str, pd.DataFrame] = {}
        for metric_type, group in df.groupby("metric_type"):
            if metric_type not in metric_types:
                continue

            pivot = group.pivot_table(
                index=["dataset", "model", "explainer"],
                columns="metric_name",
                values="metric_value",
            )

            if pivot.shape[1] < 2:
                logger.info(
                    f"Skipping correlation for '{metric_type}': "
                    f"only {pivot.shape[1]} metric(s)."
                )
                continue

            correlations[metric_type] = pivot.corr()
            logger.info(
                f"Computed {pivot.shape[1]}x{pivot.shape[1]} correlation matrix "
                f"for '{metric_type}'"
            )

        return correlations

    def draw_metric_correlation_heatmaps(
        self,
        corrs: dict[str, pd.DataFrame],
        output_path: str | Path = "correlation_heatmaps.png",
    ):
        from pathlib import Path

        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        if not corrs:
            raise ValueError("No correlation matrices provided.")

        # Convert pandas → R
        r_corrs = {}
        with localconverter(robjects.default_converter + pandas2ri.converter):
            for name, df in corrs.items():
                r_corrs[name] = robjects.conversion.py2rpy(df)

        # Push to R environment
        for name, r_df in r_corrs.items():
            robjects.globalenv[f"{name}_corr"] = r_df

        output_path = Path(output_path)
        robjects.globalenv["output_dir"] = str(output_path.parent)
        robjects.globalenv["output_stem"] = output_path.stem

        keys = list(corrs.keys())

        # Build R code to convert all correlations to long format
        long_format_code = "\n            ".join(
            [f"{k}_long <- to_long({k}_corr)" for k in keys]
        )

        # Build R code to create all plots
        plots_code = "\n            ".join(
            [
                f'p{i + 1} <- build_heatmap({k}_long, "{k.capitalize()}")'
                for i, k in enumerate(keys)
            ]
        )

        # Build R code to save individual plots
        save_code = "\n            ".join(
            [
                f'ggsave(file.path(output_dir, paste0(output_stem, "_{k}.png")), plot = p{i + 1}, width = 8, height = 6, dpi = 300)'
                for i, k in enumerate(keys)
            ]
        )

        robjects.r(f"""
            library(ggplot2)
            library(patchwork)
            library(RColorBrewer)
            library(viridis)

            to_long <- function(corr_mat) {{
                long_df <- as.data.frame(as.table(as.matrix(corr_mat)))
                colnames(long_df) <- c("Metric_X", "Metric_Y", "Correlation")
                long_df
            }}

            {long_format_code}

            build_heatmap <- function(df, title_text) {{
                ggplot(df, aes(x = Metric_X, y = Metric_Y, fill = Correlation)) +
                    geom_tile(color = "grey65", linewidth = 0.4) +
                    geom_text(aes(label = round(Correlation, 2)),
                              color = "white", size = 2) +
                    scale_fill_viridis(discrete = FALSE, direction = -1, limits = c(-1, 1), option = "magma") +
                    coord_equal() +
                    labs(title = title_text) +
                    theme_minimal(base_size = 14) +
                    theme(
                        axis.title = element_blank(),
                        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
                        plot.title = element_text(face = "bold", hjust = 0.5),
                        legend.position = "right"
                    )
            }}

            {plots_code}

            {save_code}
        """)

    # ------------------------------------------------------------------
    # Analysis: ranking
    # ------------------------------------------------------------------

    def compute_explainer_rankings(
        self,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Rank explainers per metric based on average metric value across
        dataset/model pairs.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-filtered DataFrame. If None, uses full aggregation.
        maximize : bool
            If True (default), higher metric values get lower (better) ranks.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame with columns: metric_type, metric_name,
            explainer, rank.
        """
        if df is None:
            df = self.aggregate()
        if df.empty:
            return pd.DataFrame()

        tidy_ranks: list[dict] = []

        # exclude diagnostics metrics
        df = df[df["metric_type"] != "diagnostics"]

        # Order by metric type: axiomatic, complexity, faithfulness, robustness
        type_order = {
            "axiomatic": 0,
            "complexity": 1,
            "faithfulness": 2,
            "robustness": 3,
        }
        df["metric_type_order"] = df["metric_type"].map(type_order)
        df = df.sort_values(["metric_type_order", "metric_name"]).drop(
            "metric_type_order", axis=1
        )

        explainer_order = None
        for metric_name, group in df.groupby("metric_name"):
            pivot_df = group.pivot(
                index=["dataset", "model"], columns="explainer", values="metric_value"
            )

            # print all nan rows
            nan_rows = pivot_df[pivot_df.isna().any(axis=1)]
            if not nan_rows.empty:
                logger.warning(
                    f"Found {len(nan_rows)} rows with NaN values in pivot table for metric '{metric_name}'. "
                    "This may indicate missing metrics or incomplete runs. "
                    "Affected rows:\n" + str(nan_rows)
                )

            assert not pivot_df.isna().any().any(), (
                f"Missing values in pivot table for metric '{metric_name}'. "
                "Ensure all (dataset, model, explainer) combinations are present."
            )

            metric_type = group["metric_type"].iloc[0]

            # Rank explainers WITHIN each (dataset, model) pair
            rank_df = pivot_df.rank(axis=1, method="average", ascending=False)
            explainer_order = (
                rank_df.columns if explainer_order is None else explainer_order
            )

            # Store per-(dataset, model) ranks for each explainer
            for explainer in rank_df.columns:
                for (dataset, model), rank_val in rank_df[explainer].items():
                    tidy_ranks.append(
                        {
                            "metric_type": metric_type,
                            "metric_name": metric_name,
                            "dataset": dataset,
                            "model": model,
                            "explainer": explainer,
                            "rank": rank_val,
                        }
                    )

        print("explainer_order", explainer_order)
        tidy_df = pd.DataFrame(tidy_ranks)
        logger.info(
            f"Computed rankings: {tidy_df['metric_name'].nunique()} metrics, "
            f"{tidy_df['explainer'].nunique()} explainers"
        )
        summary_by_type = (
            tidy_df.groupby(["metric_type", "explainer"])["rank"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_rank", "std": "std_rank", "count": "n"})
        )
        summary_by_type["se_rank"] = summary_by_type["std_rank"] / np.sqrt(
            summary_by_type["n"]
        )

        # reorder explainers in summary_by_type according to explainer_order
        if explainer_order is not None:
            summary_by_type["explainer"] = pd.Categorical(
                summary_by_type["explainer"],
                categories=explainer_order,
                ordered=True,
            )
            summary_by_type = summary_by_type.sort_values("explainer")

        return summary_by_type

    # ------------------------------------------------------------------
    # Analysis: critical difference diagrams
    # ------------------------------------------------------------------

    def compute_critical_difference_data(
        self,
        df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Prepare data for critical difference diagrams.

        Returns
        -------
        cd_data : np.ndarray
            Shape (n_metrics, n_dataset_model_pairs, n_explainers).
        diagram_names : list[str]
            Metric names in the same order as axis 0.
        treatment_names : list[str]
            Explainer names in column order.
        """
        if df is None:
            df = self.aggregate()
        if df.empty:
            return np.array([]), [], []

        cd_arrays: list[np.ndarray] = []
        diagram_names: list[str] = []
        treatment_names: list[str] | None = None

        # remove diagnostics metrics
        df = df[df["metric_type"] != "diagnostics"]

        # Order by metric type: axiomatic, complexity, faithfulness, robustness
        type_order = {
            "axiomatic": 0,
            "complexity": 1,
            "faithfulness": 2,
            "robustness": 3,
        }
        df["metric_type_order"] = df["metric_type"].map(type_order)
        df = df.sort_values(["metric_type_order", "metric_name"]).drop(
            "metric_type_order", axis=1
        )

        for metric_name, group in df.groupby("metric_name", sort=False):
            if metric_name in ["AOPC"]:
                logger.info(
                    f"Skipping metric '{metric_name}' for CD diagram: not a single score per (dataset, model, explainer)."
                )
                continue
            pivot = group.pivot(
                index=["dataset", "model"],
                columns="explainer",
                values="metric_value",
            )
            # see if pivot has nan rows
            if pivot.isna().any().any():
                nan_rows = pivot[pivot.isna().any(axis=1)]
                logger.warning(
                    f"Found {len(nan_rows)} rows with NaN values in pivot table for metric '{metric_name}'. "
                    "This may indicate missing metrics or incomplete runs. "
                    "Affected rows:\n" + nan_rows
                )
                exit()

            print("pivot columns", pivot)
            cd_arrays.append(pivot.to_numpy().astype(float))
            diagram_names.append(metric_name)
            treatment_names = list(pivot.columns)

        cd_data = np.stack(cd_arrays, axis=0)
        logger.info(
            f"Prepared CD data: {cd_data.shape[0]} metrics, "
            f"{cd_data.shape[1]} dataset-model pairs, "
            f"{cd_data.shape[2]} explainers"
        )
        return cd_data, diagram_names, treatment_names or []

    def generate_critical_difference_diagram(
        self,
        df: pd.DataFrame | None = None,
        output_path: str | Path = "all_metrics_cd_diagram.pdf",
        alpha: float = 0.05,
        adjustment: str = "holm",
    ) -> None:
        """Generate and save a critical difference diagram PDF.

        Requires the ``critdd`` package.
        """

        cd_data, diagram_names, treatment_names = self.compute_critical_difference_data(
            df
        )

        if cd_data.size == 0:
            logger.warning("No data for critical difference diagram.")
            return

        diagrams = MyDiagrams(
            cd_data,
            treatment_names=treatment_names,
            maximize_outcome=True,
            diagram_names=diagram_names,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = min(len(treatment_names), len(colors))
        preamble_lines = [
            f"\\definecolor{{color{i + 1}}}{{HTML}}{{{colors[i]}}}" for i in range(n)
        ]
        cycle_list = ",".join(f"{{color{i + 1},mark={markers[i]}}}" for i in range(n))
        diagrams.to_file(
            str(output_path),
            alpha=alpha,
            adjustment=adjustment,
            reverse_x=True,
            preamble="\n".join(preamble_lines),
            axis_options={
                "cycle list": cycle_list,
                "width": "\\axisdefaultwidth",
                "height": "2.0*\\axisdefaultheight",
                "title": "Explainer Rankings by Metric",
            },
        )
        logger.info(f"Saved critical difference diagram: {output_path}")

    def generate_critical_difference_diagram_with_avg_ranks(
        self,
        df: pd.DataFrame,
        output_path: str | Path = "all_metrics_cd_diagram.pdf",
    ) -> None:
        """Generate and save a critical difference diagram PDF.

        Requires the ``critdd`` package.
        """
        from critdd import tikz, tikz_2d

        cd_arrays: list[np.ndarray] = []
        diagram_names: list[str] = []
        treatment_names: list[str] = []

        for metric_type, group in df.groupby("metric_type"):
            res = group.set_index("explainer")["avg_rank"]
            cd_arrays.append(res.to_numpy())
            diagram_names.append(metric_type)
            treatment_names = list(res.index)

        cd_data = np.stack(cd_arrays, axis=0)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = min(len(treatment_names), len(colors))
        preamble_lines = [
            f"\\definecolor{{color{i + 1}}}{{HTML}}{{{colors[i]}}}" for i in range(n)
        ]
        cycle_list = ",".join(f"{{color{i + 1},mark={markers[i]}}}" for i in range(n))
        tikz.to_file(
            str(output_path),
            tikz_2d.to_str(
                cd_data,
                [[] for _ in diagram_names],
                treatment_names,
                diagram_names,
                as_document=True,
                reverse_x=True,
                preamble="\n".join(preamble_lines),
                axis_options={
                    "cycle list": cycle_list,
                    "width": "\\axisdefaultwidth",
                    "height": "\\axisdefaultheight",
                    "title": "Average Ranks by Metric Type",
                },
            ),
        )
        logger.info(f"Saved critical difference diagram: {output_path}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_to_csv(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        output_dir: str | Path | None = None,
        prefix: str = "",
        also_save_to_tmp: bool = True,
    ) -> list[Path]:
        """Save a DataFrame or dict of DataFrames to CSV."""
        if output_dir is None:
            output_dir = self._base_dir / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            data = {"data": data}

        saved: list[Path] = []
        for name, frame in data.items():
            filename = f"{prefix}_{name}.csv" if prefix else f"{name}.csv"
            filename = filename.lstrip("_")

            csv_path = output_dir / filename
            frame.to_csv(csv_path)
            saved.append(csv_path)
            logger.info(f"Saved: {csv_path}")

            if also_save_to_tmp:
                tmp_path = Path("/tmp") / filename
                frame.to_csv(tmp_path)
                logger.info(f"Saved to {tmp_path} for R plotting")

        return saved

    # ------------------------------------------------------------------
    # Run modes
    # ------------------------------------------------------------------

    def run(
        self,
        mode: Literal[
            "metric_table",
            "runtime",
            "correlations",
            "rankings",
            "critical_difference",
            "modality_fraction_comp",
            "export_all",
            "attribution_locality",
        ],
        output_dir: str | Path | None = None,
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        explainers: list[str] | None = None,
        apply_display_names: bool = True,
        **kwargs,
    ) -> None:
        """Run a summarization mode.

        Parameters
        ----------
        mode : str
            'metric_table'          — wide pivot of all metrics.
            'runtime'               — mean execution time per metric.
            'correlations'          — per-type correlation matrices.
            'rankings'              — explainer rankings per metric.
            'critical_difference'   — CD diagram PDF.
            'export_all'            — everything above.
        output_dir : str or Path, optional
            Override default output location.
        datasets, models, explainers : list[str], optional
            Filters forwarded to aggregate().
        apply_display_names : bool
            If True (default), applies DisplayConfig renaming and ordering
            before computing. Set False to keep raw names.
        **kwargs
            Forwarded to the underlying compute / generate methods.
        """
        df = self.aggregate(datasets=datasets, models=models, explainers=explainers)
        label_wise_df = self.aggregate_label_wise(
            datasets=datasets, models=models, explainers=explainers
        )

        if df.empty:
            logger.warning("Nothing to summarize — no data found.")
            return

        if apply_display_names:
            df = self.with_display_names(df)
            label_wise_df = self.with_display_names(label_wise_df)

        run_all = mode == "export_all"
        if mode == "metric_table" or run_all:  # done
            table = self.compute_metric_table(df)
            self.save_to_csv(table, output_dir=output_dir, prefix="metric_table")

        if mode == "aopc_curves" or run_all:
            from ._helpers import plot_aopc_curves

            aopc_data = df[df["metric_name"] == "AOPC"]
            plot_aopc_curves(
                aopc_data, Path(output_dir or self._base_dir / "summaries")
            )

        if mode == "runtime" or run_all:  # done
            runtime_df = self.compute_runtime_summary(df)
            self.save_to_csv(
                runtime_df, output_dir=output_dir, prefix="runtime_summary"
            )

            # draw runtime bar plot
            self.draw_runtime_bar_plot(
                runtime_df,
                output_path=Path(output_dir or self._base_dir / "summaries")
                / "runtime_summary_plot.png",
            )

        if mode == "correlations" or run_all:  # done
            df = df[df["metric_name"] != "AOPC"]
            corrs = self.compute_metric_correlations(df, **kwargs)
            self.save_to_csv(corrs, output_dir=output_dir, prefix="corr_matrix")

            # draw metric correlations
            self.draw_metric_correlation_heatmaps(
                corrs,
                output_path=Path(output_dir or self._base_dir / "summaries")
                / "correlation_heatmaps.png",
            )

        if mode == "rankings" or run_all:
            df = df[df["metric_name"] != "AOPC"]
            rankings = self.compute_explainer_rankings(df, **kwargs)
            self.generate_critical_difference_diagram_with_avg_ranks(
                rankings,
                output_path=Path(output_dir or self._base_dir / "summaries")
                / "avg_ranks_cd_diagram.pdf",
            )
            self.save_to_csv(
                rankings, output_dir=output_dir, prefix="ranking_by_metric_tidy"
            )

        if mode == "critical_difference" or run_all:
            cd_output = Path(output_dir or self._base_dir / "summaries")
            self.generate_critical_difference_diagram(
                df=df,
                output_path=cd_output / "all_metrics_cd_diagram.pdf",
                **kwargs,
            )

        if mode == "modality_fraction_comp" or run_all:
            saved = self.plot_modality_fraction_bar(
                df=df,
                label_wise_df=label_wise_df,
                output_path=Path(output_dir or self._base_dir / "summaries")
                / "modality_fraction_bar.png",
                models=["LayoutLMv3"],
                explainers=["Occ.", "IG", "RA"],
                # explainers=["Sal."],
            )
            latex = generate_latex_figure(saved, caption="Modality fraction per label.")

            # Render the latex figure to PDF
            output_dir_path = Path(output_dir or self._base_dir / "summaries")
            render_latex_to_pdf(
                latex,
                output_pdf=output_dir_path / "modality_fraction_figure.pdf",
                image_dir=output_dir_path,
            )

        if mode == "modality_frac_comp_table" or run_all:
            combined = self._prepare_modality_fraction_data(
                df=df,
                label_wise_df=label_wise_df,
                models=["BERT", "RoBERTa", "LayoutLMv3", "LiLT"],
                explainers=["Occ.", "IG", "RA"],
                seed=10,
            )

            # only get the overall rows
            overall = combined[combined["label"] == "Overall"].copy()
            print("overall", overall)
            overall = overall.pivot_table(
                index=["dataset", "model", "explainer"],
                columns="modality",
                values="score",
            ).reset_index()

            # save to csv
            overall.to_csv(
                Path(output_dir or self._base_dir / "summaries")
                / "modality_fraction_comparison_table.csv",
                index=False,
            )
            print(
                "Saved modality fraction comparison table:",
                Path(output_dir or self._base_dir / "summaries")
                / "modality_fraction_comparison_table.csv",
            )

            # Format values
            modalities = ["token", "position", "layout", "image"]
            for col in modalities:
                if col in overall.columns:
                    overall[col] = overall[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                    )

            # Sort and set index
            overall = overall.sort_values(["dataset", "model", "explainer"])
            overall = overall.set_index(["dataset", "model", "explainer"])[modalities]
            overall.columns = ["Token", "Position", "Layout", "Image"]

            # Dump as LaTeX
            latex_str = overall.to_latex(
                multirow=True,
                caption="Overall modality contribution fractions across all datasets for \\layoutlm{} and \\lilt{}.",
                label="tab:modality_fractions",
                column_format="lll cccc",
                escape=False,
            )

            print(latex_str)
            latex_str = latex_str.replace("cline", "cmidrule")

            with open(
                Path(output_dir or self._base_dir / "summaries")
                / "modality_fraction_comparison_table.tex",
                "w",
            ) as f:
                f.write(latex_str)

        if mode == "attribution_locality" or run_all:
            scalars_df = extract_diagnostic_scalars(
                df,
                models=["LayoutLMv3"],
                explainers=["Occ.", "IG", "RA"],
            )
            label_wise_scalars_df = extract_diagnostic_scalars(
                label_wise_df,
                models=["LayoutLMv3"],
                explainers=["Occ.", "IG", "RA"],
            )
            label_wise_scalars_df = collapse_labels(
                label_wise_scalars_df, ["x_locality", "y_locality", "spread"]
            )
            label_wise_scalars_df = select_top_labels(
                label_wise_scalars_df,
                ["x_locality", "y_locality", "spread"],
                top_k=4,
                remove_O=False,
            )
            # see if all rows nan we ignore it
            scalars_df.to_csv(
                Path(output_dir or self._base_dir / "summaries")
                / "diagnostic_scalars.csv",
                index=False,
            )
            label_wise_scalars_df.to_csv(
                Path(output_dir or self._base_dir / "summaries")
                / "diagnostic_scalars_label_wise.csv",
                index=False,
            )
            if not scalars_df["x_locality"].isna().all():
                draw_locality_confusion(
                    scalars_df,
                    label_wise_scalars_df,
                    output_path=Path(output_dir or self._base_dir / "summaries")
                    / "diagnostic_scalars.png",
                )

        if mode == "topk_word_freq" or run_all:
            from ._helpers import draw_word_freq_bars, extract_topk_word_freq

            label_wise_topk_word_freq_df = extract_topk_word_freq(
                label_wise_df,
                models=["LayoutLMv3"],
                explainers=["Occ."],
                top_n=6,
            )
            label_wise_topk_word_freq_df = select_top_labels(
                label_wise_topk_word_freq_df,
                ["frequency"],
                top_k=6,
                remove_O=True,
            )

            draw_word_freq_bars(
                df=label_wise_topk_word_freq_df,
                output_dir=Path(output_dir or self._base_dir / "summaries"),
            )


def generate_latex_figure(
    saved_paths: list[str | Path],
    caption: str = "Modality fraction heatmaps.",
    label: str = "fig:modality_fraction",
) -> str:
    """Generate LaTeX code that stacks saved figures vertically into one float.

    Parameters
    ----------
    saved_paths : list of paths returned by ``plot_modality_fraction_bar``
    caption : overall figure caption
    label : LaTeX label for cross-referencing

    Returns
    -------
    str : LaTeX code ready to paste into a document.
    """
    from pathlib import Path

    subfigs = []
    for p in saved_paths:
        p = Path(p)
        subfigs.append(
            f"    \\begin{{subfigure}}{{\\textwidth}}\n"
            f"        \\centering\n"
            f"        \\includegraphics[width=\\textwidth]{{{p.name}}}\n"
            f"    \\end{{subfigure}}"
        )

    body = "\n".join(subfigs)

    latex = (
        f"\\begin{{figure}}[htbp]\n"
        f"    \\centering\n"
        f"{body}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        f"\\end{{figure}}"
    )
    return latex


def render_latex_to_pdf(
    latex_body: str,
    output_pdf: str | Path,
    image_dir: str | Path | None = None,
) -> Path:
    """Render a LaTeX figure snippet to a standalone PDF using pdflatex.

    Parameters
    ----------
    latex_body : str
        LaTeX code (figure environment) to render.
    output_pdf : str or Path
        Destination PDF path.
    image_dir : str or Path, optional
        Directory added to ``\\graphicspath`` so ``\\includegraphics`` can
        find the image files.

    Returns
    -------
    Path
        Resolved path to the generated PDF.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    graphicspath = ""
    if image_dir is not None:
        gp = Path(image_dir).resolve()
        graphicspath = f"\\graphicspath{{{{{gp}/}}}}"

    document = (
        "\\documentclass{article}\n"
        "\\usepackage[margin=1cm]{geometry}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{subcaption}\n"
        "\\usepackage{caption}\n"
        "\\pagestyle{empty}\n"
        f"{graphicspath}\n"
        "\\begin{document}\n"
        f"{latex_body}\n"
        "\\end{document}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "figure.tex"
        tex_file.write_text(document)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tex_file)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdflatex failed:\n{result.stdout}\n{result.stderr}")

        generated_pdf = Path(tmpdir) / "figure.pdf"

        # Crop to tight bounding box if pdfcrop is available
        crop_result = subprocess.run(
            ["pdfcrop", str(generated_pdf), str(generated_pdf)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        if crop_result.returncode != 0:
            logger.warning(
                "pdfcrop not available or failed; PDF will have full page margins."
            )

        shutil.copy2(generated_pdf, output_pdf)

    logger.info(f"Saved LaTeX figure PDF: {output_pdf}")
    return output_pdf
