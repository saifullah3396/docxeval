from __future__ import annotations

import ast
import json
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))


def is_valid_word(word: str) -> bool:
    word = word.lower().strip()
    if word in STOP_WORDS:
        return False
    if len(word) < 2:
        return False
    return True


def extract_diagnostic_scalars(
    df: pd.DataFrame,
    metric_name: str = "WordAttributionLocality",  # or whatever the metric_name is for these rows
    value_col: str = "metric_value",
    scalar_keys: list[str] | None = None,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    explainers: list[str] | None = None,
) -> pd.DataFrame:
    """Extract x_locality, y_locality, spread (or any scalar keys) into a flat df.

    Returns
    -------
    DataFrame with columns: dataset, model, explainer, + one col per scalar key
    """
    if scalar_keys is None:
        scalar_keys = ["x_locality", "y_locality", "spread"]

    sub = df[df["metric_name"] == metric_name].copy()

    if datasets:
        sub = sub[sub["dataset"].isin(datasets)]
    if models:
        sub = sub[sub["model"].isin(models)]
    if explainers:
        sub = sub[sub["explainer"].isin(explainers)]

    records = []
    for _, row in sub.iterrows():
        d = row[value_col] if isinstance(row[value_col], dict) else {}
        updated_row = {
            "dataset": row["dataset"],
            "model": row["model"],
            "explainer": row["explainer"],
            **{k: d.get(k, np.nan) for k in scalar_keys},
        }
        if "label" in row:
            updated_row["label"] = row["label"]
        records.append(updated_row)

    return pd.DataFrame(records)


def collapse_labels(df: pd.DataFrame, scalar_keys: list[str]) -> pd.DataFrame:
    if "label" not in df.columns:
        return df

    df = df.copy()
    df["label"] = df["label"].str.replace(r"^[BI]-", "", regex=True)

    return df.groupby(["dataset", "model", "explainer", "label"], as_index=False)[
        scalar_keys
    ].mean()


def select_top_labels(
    df: pd.DataFrame,
    scalar_keys: list[str],
    top_k: int = 4,
    remove_O: bool = True,
) -> pd.DataFrame:
    import pandas as pd

    if "label" not in df.columns:
        return df

    out = []

    for (ds, mdl), sub in df.groupby(["dataset", "model"]):
        # long format for variance calc
        tmp = sub.melt(
            id_vars=["dataset", "model", "explainer", "label"],
            value_vars=scalar_keys,
            var_name="metric",
            value_name="value",
        )

        # compute variance per label
        scores = (
            tmp.groupby(["label", "metric"])["value"]
            .std()
            .groupby("label")
            .mean()
            .sort_values(ascending=False)
        )

        labels = list(scores.index)

        if remove_O:
            labels = [l for l in labels if l != "O"]
            selected = labels[:top_k]
        else:
            labels_wo_O = [l for l in labels if l != "O"]
            selected = labels_wo_O[: max(0, top_k - 1)]
            if "O" in labels:
                selected = ["O"] + selected

        out.append(sub[sub["label"].isin(selected)])

    return pd.concat(out, ignore_index=True)


def extract_topk_word_freq(
    df: pd.DataFrame,
    metric_name: str = "TextProfile",
    value_col: str = "metric_value",
    freq_key: str = "topk_word_freq",
    top_n: int = 15,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    explainers: list[str] | None = None,
) -> pd.DataFrame:
    """Extract top-k word frequencies into a long-form df.

    Returns
    -------
    DataFrame with columns: dataset, model, explainer, word, frequency
    """
    sub = df[df["metric_name"] == metric_name].copy()

    if datasets:
        sub = sub[sub["dataset"].isin(datasets)]
    if models:
        sub = sub[sub["model"].isin(models)]
    if explainers:
        sub = sub[sub["explainer"].isin(explainers)]

    records = []
    for _, row in sub.iterrows():
        d = row[value_col] if isinstance(row[value_col], dict) else {}
        word_freq = d.get(freq_key, {})
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        filtered_words = {}
        for w, f in sorted_words:
            if is_valid_word(w):
                filtered_words[w] = f
            else:
                filtered_words["STOP"] = filtered_words.get("STOP", 0) + f

        sorted_filtered = sorted(
            filtered_words.items(), key=lambda x: x[1], reverse=True
        )
        filtered_words = sorted_filtered[:top_n]
        # instead of removing stop words we just replace them all with a single "STOP" token, this way we can still see if stop words are dominating the top-k
        # print("sorted_words", sorted_words)
        # filtered_words = [
        #     (w if is_valid_word(w) else "STOP", f) for w, f in sorted_words
        # ][:top_n]
        # print("filtered_words", filtered_words)

        add_args = {}
        if "label" in row:
            add_args["label"] = row["label"]

            # if label as bio tags just remove them
            if row["label"].startswith(("B-", "I-")):
                add_args["label"] = row["label"][2:]

            # if label is "O" we can remove it
            if add_args["label"] == "O":
                continue

        for word, freq in filtered_words:
            records.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "explainer": row["explainer"],
                    "word": word,
                    "frequency": freq,
                    **add_args,
                }
            )

    return pd.DataFrame(records)


def extract_label_wise_ner(
    label_wise_df: pd.DataFrame,
    metric_name: str = "TextProfile",
    value_col: str = "metric_value",
    ner_key: str = "mean_ner",
    target_label_col: str = "label",  # adjust to your actual column name
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    explainers: list[str] | None = None,
) -> pd.DataFrame:
    """Extract label-to-label NER weights into a long-form df.

    Returns
    -------
    DataFrame with columns: dataset, model, explainer, target_label, source_label, weight
    """
    sub = label_wise_df[label_wise_df["metric_name"] == metric_name].copy()

    if datasets:
        sub = sub[sub["dataset"].isin(datasets)]
    if models:
        sub = sub[sub["model"].isin(models)]
    if explainers:
        sub = sub[sub["explainer"].isin(explainers)]

    records = []
    for _, row in sub.iterrows():
        d = row[value_col] if isinstance(row[value_col], dict) else {}
        ner_dict = d.get(ner_key, {})
        target = row[target_label_col]
        for source_label, weight in ner_dict.items():
            records.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "explainer": row["explainer"],
                    "target_label": target,
                    "source_label": source_label,
                    "weight": weight,
                }
            )

    return pd.DataFrame(records)


def draw_locality_confusion(
    scalars_df: pd.DataFrame,
    label_wise_scalars_df: pd.DataFrame,
    output_path: str | Path = "locality_confusion.png",
    scalar_keys: list[str] | None = None,
):
    from pathlib import Path

    import pandas as pd
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    if scalar_keys is None:
        scalar_keys = ["v_locality", "h_locality", "spread"]

    output_path = Path(output_path)

    # ---- 1. Prepare data ------------------------------------------------------

    # Overall → add label
    overall = scalars_df.copy()
    overall["label"] = "Overall"

    # Collapse B- / I- → single label
    lw = label_wise_scalars_df.copy()
    lw["label"] = lw["label"].str.replace(r"^[BI]-", "", regex=True)

    # rename x-locality to v-locality and y-locality to h-locality for better plot labels
    overall["v_locality"] = overall.pop("x_locality")
    overall["h_locality"] = overall.pop("y_locality")

    lw["v_locality"] = lw.pop("x_locality")
    lw["h_locality"] = lw.pop("y_locality")

    # Aggregate after collapsing
    lw = lw.groupby(["dataset", "model", "explainer", "label"], as_index=False)[
        scalar_keys
    ].mean()

    combined = pd.concat([overall, lw], ignore_index=True)

    # Pivot to long
    combined = combined.melt(
        id_vars=["dataset", "model", "explainer", "label"],
        value_vars=scalar_keys,
        var_name="metric",
        value_name="value",
    )

    # ---- 2. Loop per dataset-model -------------------------------------------
    combos = combined[["dataset", "model"]].drop_duplicates().values.tolist()

    for ds, mdl in combos:
        sub = combined[(combined["dataset"] == ds) & (combined["model"] == mdl)].copy()

        labels = sorted(sub["label"].unique())
        if "Overall" in labels:
            labels.remove("Overall")
        labels = ["Overall"] + labels

        n_labels = len(labels)

        import itertools

        explainers = sorted(sub["explainer"].unique())
        metrics = scalar_keys

        full_grid = pd.DataFrame(
            list(itertools.product(labels, explainers, metrics)),
            columns=["label", "explainer", "metric"],
        )

        sub = full_grid.merge(sub, on=["label", "explainer", "metric"], how="left")

        out_path = output_path.with_name(
            f"{output_path.stem}_{ds}_{mdl}{output_path.suffix}"
        )

        plot_defs = []
        for j, lbl in enumerate(labels):
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

            metric_levels <- c({", ".join(f'"{m}"' for m in scalar_keys)})
            plot_df$metric <- factor(plot_df$metric, levels = metric_levels)

            build_heatmap <- function(df, title_text) {{
                df$text_label <- ifelse(is.na(df$value),
                                        "N/A",
                                        sprintf("%.2f", df$value))

                df$text_color <- ifelse(is.na(df$value), "na",
                                 ifelse(df$value > 0.5, "high", "low"))

                ggplot(df, aes(x = metric, y = explainer)) +
                    geom_tile(aes(fill = value),
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
                                                   size = 8),
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
                    title = "{ds} / {mdl}"
                ) &
                theme(legend.position = "right")

            fig_w <- max(5, {n_labels} * 2.8)
            fig_h <- 3

            ggsave(
                "{str(out_path)}",
                plot = p,
                width = fig_w,
                height = fig_h,
                dpi = 300
            )
        """

        with localconverter(robjects.default_converter + pandas2ri.converter):
            robjects.globalenv["plot_df"] = robjects.conversion.py2rpy(sub)

        robjects.r(r_script)

    print("Saved locality confusion plots.")


from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def draw_word_freq_bars(
    df: pd.DataFrame,
    output_dir: str | Path = "word_freq_plots",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r = robjects.r

    # Load required libraries once (avoids repeated initialization instability)
    r("""
    suppressPackageStartupMessages({
        library(ggplot2)
        library(dplyr)
    })
    """)

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds].copy()

        # Convert pandas -> R safely
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.conversion.py2rpy(sub)

        out_path = output_dir / f"{ds}_word_freq.png"

        # Push objects into R global env
        robjects.globalenv["df"] = r_df
        robjects.globalenv["output_path"] = str(out_path)

        r("""
df <- as.data.frame(df)

df <- df %>%
    group_by(label, word) %>%
    summarise(freq = sum(frequency, na.rm = TRUE), .groups = "drop")

df <- df %>%
    group_by(label) %>%
    mutate(
        word = reorder(word, freq),
        label_total = sum(freq, na.rm = TRUE),
        y_label = interaction(label, word, sep = " :: ")
    ) %>%
    ungroup()

df$label <- reorder(df$label, df$label_total)

df$is_stop <- grepl("STOP", df$word)

n_labels <- length(unique(df$label))

# set riw as 1 if less than 4 labels, otherwise set it to 2
legend_nrow <- ifelse(n_labels <= 4, 1, 2)

p <- ggplot(df, aes(x = freq, y = y_label, fill = label, linetype = is_stop)) +
    geom_col(width = 0.7, color = "black", linewidth = 0.3, aes(fill = label)) +
    geom_col(data = df[df$is_stop, ], width = 0.7, color = "black", linewidth = 0.3, fill = NA, linetype = "dashed") +

    scale_y_discrete(labels = function(x) sub(".* :: ", "", x)) +
    scale_x_continuous(position = "top", expand = c(0, 0)) +
    scale_fill_brewer(palette = "Set2") +
    scale_linetype_manual(values = c("FALSE" = "solid", "TRUE" = "dotted"), guide = "none") +

    labs(
        x = "Frequency",
        y = NULL,
        fill = "Label"
    ) +

    theme(
        axis.title.y = element_blank(),
        axis.text.y = element_text(size = 4, color = "black"),
        axis.text.x = element_text(size = 10, color = "black"),

        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),

        legend.position = "top",
        legend.title = element_blank(),

        axis.line = element_line(color = "black", linewidth = 0.3),
        text = element_text(size = 10),
    ) +

    guides(
        fill = guide_legend(
            nrow = legend_nrow,
            byrow = TRUE,
            direction = "horizontal",
            override.aes = list(color = "black"),
        )
    )

n_total_labels <- nrow(df)
height <- max(3, n_total_labels * 0.08)

ggsave(
    filename = output_path,
    plot = p,
    width = 5,
    height = height,
    dpi = 300,
)
""")

        print(f"Saved: {out_path}")


def draw_ner_heatmap(
    self,
    ner_df: pd.DataFrame,
    output_path: str | Path = "ner_label_heatmap.png",
):
    """Heatmap of mean NER attribution weights: target_label x source_label.

    One panel per explainer, faceted by dataset-model on rows.
    """
    from pathlib import Path

    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_ner = robjects.conversion.py2rpy(ner_df)
        robjects.globalenv["ner_df"] = r_ner

    robjects.globalenv["output_path"] = str(Path(output_path))

    robjects.r("""
        library(ggplot2)
        library(dplyr)
        library(viridis)

        ner_df <- ner_df %>%
            mutate(facet_label = paste(dataset, model, sep = " / "))

        p <- ggplot(ner_df, aes(x = source_label, y = target_label, fill = weight)) +
            geom_tile(color = "grey65", linewidth = 0.3) +
            scale_fill_viridis(discrete = FALSE, option = "magma", direction = -1) +
            facet_grid(facet_label ~ explainer) +
            coord_equal() +
            labs(x = "Source label", y = "Target label", fill = "Weight") +
            theme_minimal(base_size = 12) +
            theme(
                axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
                strip.text = element_text(face = "bold"),
                plot.title = element_text(face = "bold", hjust = 0.5),
                legend.position = "right"
            )

        n_facet_rows <- length(unique(ner_df$facet_label))
        n_facet_cols <- length(unique(ner_df$explainer))

        ggsave(
            output_path,
            plot = p,
            width = 3 * n_facet_cols + 2,
            height = 3 * n_facet_rows + 1,
            dpi = 300,
        )

        p
    """)


def _parse_metric_value(val):
    """Safely parse metric_value whether it's already a dict or a string."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val.replace("'", '"'))
        except json.JSONDecodeError:
            return ast.literal_eval(val)
    return val


def _explode_aopc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the compact DataFrame (one row per dataset-model-explainer with
    metric_value dict) into long-form, subtracting the random baseline
    element-wise from desc and asc.

    Returns columns:
        dataset, model, explainer, curve_type, step, aopc_value, pct_removed
    """
    rows = []

    dataset_order = {ds: i for i, ds in enumerate(df["dataset"].unique())}
    model_order = {mdl: i for i, mdl in enumerate(df["model"].unique())}
    explainer_order = {exp: i for i, exp in enumerate(df["explainer"].unique())}

    for _, row in df.iterrows():
        mv = _parse_metric_value(row["metric_value"])

        dataset = row["dataset"]
        model = row["model"]
        explainer = row["explainer"]

        desc_vals = mv.get("aopc.desc") or []
        asc_vals = mv.get("aopc.asc") or []
        rand_vals = mv.get("aopc.random") or []
        for curve_label, vals in [("desc", desc_vals), ("asc", asc_vals)]:
            if not vals:
                continue
            max_step = len(vals) - 1
            for i, v in enumerate(vals):
                if v is None:
                    continue
                r = rand_vals[i]
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "explainer": explainer,
                        "curve_type": curve_label,
                        "step": i,
                        "aopc_value": float(v) - float(r),
                        "pct_removed": (i / max_step * 100) if max_step > 0 else 0.0,
                    }
                )
    df = pd.DataFrame(rows)

    # reorder categories for consistent plotting
    df["dataset"] = pd.Categorical(
        df["dataset"], categories=sorted(dataset_order, key=dataset_order.get)
    )
    df["model"] = pd.Categorical(
        df["model"], categories=sorted(model_order, key=model_order.get)
    )
    df["explainer"] = pd.Categorical(
        df["explainer"], categories=sorted(explainer_order, key=explainer_order.get)
    )
    return df


def plot_aopc_curves(
    aopc_df: pd.DataFrame,
    output_dir: str | Path = "./plots",
):
    """
    Produce a single AOPC-curve PNG faceted by dataset (rows) and model (cols).

    Parameters
    ----------
    aopc_df : pd.DataFrame
        Must contain columns: dataset, model, explainer, metric_value
    output_dir : str | Path
        Directory where the PNG is saved (created if needed).
    """
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Python-side preprocessing -------------------------------------------
    long_df = _explode_aopc(aopc_df)

    # Split into curve types (already baseline-subtracted, pct_removed computed)
    desc_df = long_df[long_df["curve_type"] == "desc"].copy()
    asc_df = long_df[long_df["curve_type"] == "asc"].copy()

    # Push DataFrames to R
    with localconverter(robjects.default_converter + pandas2ri.converter):
        robjects.globalenv["desc_df"] = robjects.conversion.py2rpy(desc_df)
        robjects.globalenv["asc_df"] = robjects.conversion.py2rpy(asc_df)

    n_datasets = long_df["dataset"].nunique()
    n_models = long_df["model"].nunique()
    plot_width = max(7 * n_models, 14)
    plot_height = max(6 * n_datasets, 10)

    out_path = str(output_dir / "aopc_curves.png")
    robjects.globalenv["out_path"] = out_path
    robjects.globalenv["plot_width"] = plot_width
    robjects.globalenv["plot_height"] = plot_height

    robjects.r(
        """
        library(ggplot2)
        library(dplyr)
        library(viridis)

        # ── Method classification ───────────────────────────────
        gradient_methods <- c("Saliency", "DeepLift", "DeepLiftShap",
                              "GradientShap", "InputXGradient",
                              "IntegratedGradients")
        perturbation_methods <- c("FeatureAblation", "Lime",
                                  "KernelShap", "Occlusion")

        classify <- function(df) {
            df %>%
                mutate(
                    method_type = case_when(
                        explainer %in% gradient_methods     ~ "Gradient-based",
                        explainer %in% perturbation_methods ~ "Perturbation-based",
                        TRUE ~ "Other"
                    ),
                    explainer = factor(
                        explainer,
                        levels = c(gradient_methods,
                                   perturbation_methods,
                                   setdiff(unique(explainer),
                                           c(gradient_methods,
                                             perturbation_methods)))
                    )
                )
        }

        desc_df <- classify(desc_df)
        asc_df  <- classify(asc_df)

        # ── Build plot (desc and asc already shifted by random) ─
        p <- ggplot()

        # Descending (solid) — already baseline-subtracted
        if (nrow(desc_df) > 0) {
            p <- p + geom_line(
                data = desc_df,
                aes(x = pct_removed, y = aopc_value, color = explainer),
                linewidth = 1.2, alpha = 0.7
            )
        }

        # Ascending (dashed) — already baseline-subtracted
        if (nrow(asc_df) > 0) {
            p <- p + geom_line(
                data = asc_df,
                aes(x = pct_removed, y = aopc_value, color = explainer),
                linewidth = 1.2, alpha = 0.7, linetype = "dashed"
            )
        }

        # Facet: dataset on rows, model on columns
        p <- p +
            facet_grid(dataset ~ model, scales = "free_y") +
            labs(
                x     = "% Tokens Removed",
                y     = "AOPC",
                color = "XAI Method",
                title = "AOPC Curves"
            ) +
            theme_gray(base_size = 16) +
            theme(
                legend.position  = "bottom",
                legend.box       = "horizontal",
                legend.title     = element_text(face = "bold"),
                legend.spacing   = unit(0.5, "cm"),
                plot.title       = element_text(face = "bold", hjust = 0.5,
                                                margin = margin(b = 10)),
                axis.title       = element_text(face = "bold"),
                plot.margin      = margin(15, 15, 15, 15),
                strip.text       = element_text(face = "bold", size = 14),
                aspect.ratio     = 1
            ) +
            scale_color_manual(values = c(
                "#84B818",
                "#D18B12",
                "#1BB5B5",
                "#F85A3E",
                "#4B6CFC",
                "#E377C2",
                "#7F7F7F",
                "#BCBD22",
                "#17BECF",
                "#AEC7E8",
                "#FFBB78",
                "#FF7F0E",
                "#2CA02C"
            )) +
            guides(color = guide_legend(nrow = 2, byrow = TRUE))

        ggsave(
            filename = out_path,
            plot     = p,
            width    = plot_width,
            height   = plot_height,
            bg       = "white"
        )

        cat("Saved:", out_path, "\n")
        """
    )

    print(f"Done. Plots saved to {output_dir}/")
