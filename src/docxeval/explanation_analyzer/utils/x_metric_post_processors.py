from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel

from docxeval.explanation_analyzer.utils.metric_funcs import (
    compute_effective_complexity,
    compute_non_sens,
)


def resample_and_plot(data, target_len=101, draw=False):
    """
    Resample variable-length arrays in a DataFrame to a common 0–100% scale
    and plot before/after alignment correctly.
    """

    def resample_to_percentage(arr):
        arr = np.asarray(arr, dtype=float)
        n = len(arr)

        if n < 2:
            return np.full(target_len, arr[0])

        source_x = np.linspace(0, 1, n)
        target_x = np.linspace(0, 1, target_len)

        return np.interp(target_x, source_x, arr)

    if draw:
        plt.figure()
        for arr in data:
            plt.plot(arr)  # x = raw index

        plt.xlabel("Original Feature Index")
        plt.ylabel("Value")
        plt.title("BEFORE: Original Arrays (Different Lengths)")
        plt.show()

    data = data.apply(resample_to_percentage)

    if draw:
        plt.figure()
        x = np.linspace(0, 100, target_len)

        for arr in data:
            plt.plot(x, arr)

        plt.xlabel("Feature Percentage (%)")
        plt.ylabel("Value")
        plt.title("AFTER: Resampled Arrays (Aligned to 0–100%)")
        plt.show()

    return data


# since features can be different per sample, we take the max size
# we interpolate all samples to a size of 100 correpsonding to 100%
def resample_to_percentage(arr, target_len=101):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return np.full(target_len, arr[0])
    source_x = np.linspace(0, 1, n)
    target_x = np.linspace(0, 1, target_len)
    return np.interp(target_x, source_x, arr)


class MetricPostProcessor(BaseModel):
    metric_name: str
    column_name: Union[str, List[str]]
    exec_time_column_name: str | list[str]
    type: str
    is_lower_the_better: bool = False
    metric_perturbation_type: str | None = None
    returns_single_value: bool = True
    use_correct_predictions: bool = True
    set_nan_zero: bool = False

    def sanitize_col(
        self,
        col: pd.Series,
        is_correct_col: pd.Series | None,
        is_multi_target: bool = False,
        label: str | int | None = None,
        label_col: pd.Series | None = None,
    ) -> pd.Series:
        if is_multi_target:
            col = col.explode()
            if is_correct_col is not None:
                is_correct_col = is_correct_col.explode().astype(bool)

        if label is not None and label_col is not None:
            if is_multi_target:
                label_mask = label_col.explode() == label
            else:
                label_mask = label_col == label
            col = col[label_mask]
            if is_correct_col is not None:
                is_correct_col = is_correct_col[label_mask]

        if self.use_correct_predictions:
            if is_correct_col is None:
                raise ValueError(
                    "DataFrame does not contain 'is_correct' column for filtering."
                )
            col = col[is_correct_col]

        return col

    def sanitize_df(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,  # NEW
    ) -> pd.DataFrame:
        column_names = (
            [self.column_name]
            if isinstance(self.column_name, str)
            else self.column_name
        )
        is_correct_col = df["is_correct"] if "is_correct" in df.columns else None
        label_col = (
            df["label"] if (label is not None and "label" in df.columns) else None
        )

        sanitized_cols = {}
        for col in column_names:
            sanitized_cols[col] = self.sanitize_col(
                df[col],
                is_correct_col,
                is_multi_target=is_multi_target,
                label=label,  # NEW
                label_col=label_col,  # NEW
            )
        if self.set_nan_zero:
            for col in column_names:
                sanitized_cols[col] = sanitized_cols[col].fillna(0)

        sanitized_df = pd.DataFrame(sanitized_cols)
        if sanitized_df[column_names].isna().any().any():
            print(
                f"N rows contain NaN values in columns {column_names} after sanitization. Sample rows with NaN values:\n{sanitized_df[sanitized_df[column_names].isna().any(axis=1)].head()}"
            )
            # removethe nan rows
            sanitized_df = sanitized_df.dropna(subset=column_names)

        # assert not sanitized_df[column_names].isna().any().any(), (
        #     f"NaN values found in columns {column_names} after sanitization. rows with NaN values:\n{sanitized_df[sanitized_df[column_names].isna().any(axis=1)]}"
        # )
        return sanitized_df

    def _summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,  # NEW
    ) -> pd.Series:
        df = self.sanitize_df(df, is_multi_target=is_multi_target, label=label)
        if len(df) == 0:
            return np.nan
        return df[  # The above code is not valid Python code. The `self` keyword is typically used
            # within a class definition in Python to refer to the instance of the class itself.
            # However, in this context, it is not being used correctly. The `
            self.column_name
        ].mean()

    def summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,  # NEW
    ) -> pd.Series:
        exec_time = self.exec_time(df)
        summarized = self._summarize(df, is_multi_target=is_multi_target, label=label)
        if isinstance(summarized, np.float32):
            summarized = summarized.item()
        if isinstance(summarized, float):
            summarized = {"score": summarized}
        elif isinstance(summarized, pd.Series):
            summarized = summarized.to_dict()
        return {
            **summarized,
            **exec_time,
            "is_lower_the_better": self.is_lower_the_better,
            "metric_perturbation_type": self.metric_perturbation_type,
            "type": self.type,
        }

    def exec_time(self, df: pd.DataFrame) -> dict:
        # add per target if multi-target
        result = (
            df[self.exec_time_column_name] / df["n_targets"]
            if "n_targets" in df.columns
            else df[self.exec_time_column_name]
        )
        return {"exec_time": result.mean()}


class CompletenessPostProcessor(MetricPostProcessor):
    metric_name: str = "Completeness"
    column_name: str = "completeness.score"
    exec_time_column_name: str = "completeness.sample_exec_time"
    type: str = "axiomatic"
    is_lower_the_better: bool = True


class ComplexityEntropyPostProcessor(MetricPostProcessor):
    metric_name: str = "Entropy-Based Complexity"
    column_name: str = "complexity_entropy.score"
    exec_time_column_name: str = "complexity_entropy.sample_exec_time"
    type: str = "complexity"
    is_lower_the_better: bool = True
    metric_perturbation_type: str | None = None

    def _summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,
    ) -> pd.Series:
        metric_col = df[self.column_name]
        n_features = df["total_features"]
        is_correct_col = df["is_correct"] if "is_correct" in df.columns else None
        label_col = df["label"] if label is not None else None

        if is_multi_target:
            metric_col = metric_col.explode().astype(float)
            n_features = n_features.explode().astype(int)
            if is_correct_col is not None:
                is_correct_col = is_correct_col.explode().astype(bool)
            if label is not None:
                label_mask = label_col.explode() == label
                metric_col = metric_col[label_mask]
                n_features = n_features[label_mask]
                if is_correct_col is not None:
                    is_correct_col = is_correct_col[label_mask]
        elif label is not None:
            label_mask = label_col == label
            metric_col = metric_col[label_mask]
            n_features = n_features[label_mask]
            if is_correct_col is not None:
                is_correct_col = is_correct_col[label_mask]

        if len(metric_col) == 0:
            return np.nan

        if self.use_correct_predictions:
            if is_correct_col is None:
                raise ValueError(
                    "DataFrame does not contain 'is_correct' column for filtering."
                )
            metric_col = metric_col[is_correct_col]
            n_features = n_features[is_correct_col]

        return np.clip(metric_col / n_features.apply(np.log), 0, 1).mean()


class SundarajajanComplexityPostProcessor(MetricPostProcessor):
    metric_name: str = "Sundararajan Complexity"
    column_name: str = "complexity_s.score"
    exec_time_column_name: str = "complexity_s.sample_exec_time"
    type: str = "complexity"
    is_lower_the_better: bool = True
    metric_perturbation_type: str | None = None


class SparsenessPostProcessor(MetricPostProcessor):
    metric_name: str = "Sparseness"
    column_name: str = "sparseness.score"
    exec_time_column_name: str = "sparseness.sample_exec_time"
    type: str = "complexity"
    is_lower_the_better: bool = False
    metric_perturbation_type: str | None = None


class EffectiveComplexityPostProcessor(MetricPostProcessor):
    metric_name: str = "Effective Complexity"
    column_name: List[str] = [
        "effective_complexity.score",
        "effective_complexity.perturbed_fwd_diffs_relative_vars_batch",
    ]
    exec_time_column_name: str = "effective_complexity.sample_exec_time"
    type: str = "complexity"
    is_lower_the_better: bool = True
    metric_perturbation_type: str = "ordered_perturbation"

    def _summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        zero_variance_threshold: float = 0.01,
        label: str | int | None = None,
    ) -> pd.Series:
        fwd_diff_vars = df[self.column_name[1]]
        fwd_diff_vars = self.sanitize_col(
            fwd_diff_vars,
            df.get("is_correct", None),
            is_multi_target=is_multi_target,
            label=label,
            label_col=df["label"] if label is not None else None,
        )

        if len(fwd_diff_vars) == 0:
            return {
                "score": np.nan,
            }

        scores = fwd_diff_vars.apply(
            compute_effective_complexity,
            zero_variance_threshold=zero_variance_threshold,
            return_ratio=True,
        )
        return {
            "score": scores.mean(),
        }


class NonSensitivityPostProcessor(MetricPostProcessor):
    metric_name: str = "Non-Sensitivity"
    column_name: List[str] = [
        "monotonicity_corr_and_non_sens.non_sensitivity",
        "monotonicity_corr_and_non_sens.perturbed_fwd_diffs_relative_vars",
        "monotonicity_corr_and_non_sens.feature_group_attribution_scores",
    ]
    exec_time_column_name: str = "monotonicity_corr_and_non_sens.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = True
    metric_perturbation_type: str = "unordered_perturbation"

    def _summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        # zero_attribution_threshold: float = 1e-2,
        # zero_variance_threshold: float = 1e-2,
        zero_attribution_threshold=1.0e-3,
        zero_variance_threshold=1.0e-2,
        label: str | int | None = None,
    ) -> pd.Series:
        def compute_non_sensitive_score(row):
            perturbed_fwd_diffs_relative_vars = row[self.column_name[1]]
            feature_group_attribution_scores = row[self.column_name[2]]

            return compute_non_sens(
                perturbed_fwd_diffs_relative_vars,
                feature_group_attribution_scores,
                zero_attribution_threshold=zero_attribution_threshold,
                zero_variance_threshold=zero_variance_threshold,
                return_ratio=True,
            )

        def compute_zero_attr_non_sensitive_score(row):
            perturbed_fwd_diffs_relative_vars = row[self.column_name[1]]
            feature_group_attribution_scores = row[self.column_name[2]]

            return compute_non_sens(
                perturbed_fwd_diffs_relative_vars,
                np.zeros_like(feature_group_attribution_scores),
                zero_attribution_threshold=zero_attribution_threshold,
                zero_variance_threshold=zero_variance_threshold,
                return_ratio=True,
            )

        # original_score = df[self.column_name[0]]
        df = self.sanitize_df(df, is_multi_target=is_multi_target, label=label)

        if len(df) == 0:
            return {
                "score": np.nan,
            }

        updated_score = df.apply(lambda row: compute_non_sensitive_score(row), axis=1)
        # zero_score = df.apply(
        #     lambda row: compute_zero_attr_non_sensitive_score(row), axis=1
        # )
        return {
            "score": updated_score.mean(),
        }


class FaithfulnessCorrelation20PostProcessor(MetricPostProcessor):
    metric_name: str = "Faithfulness Correlation (20)"
    column_name: str = "faithfulness_correlation_20.faithfulness_corr_score"
    exec_time_column_name: str = "faithfulness_correlation_20.sample_exec_time"
    type: str = "faithfulness"
    metric_perturbation_type: str = "unordered_perturbation"
    is_lower_the_better: bool = False
    set_nan_zero: bool = True


class FaithfulnessCorrelation40PostProcessor(FaithfulnessCorrelation20PostProcessor):
    metric_name: str = "Faithfulness Correlation (40)"
    column_name: str = "faithfulness_correlation_40.faithfulness_corr_score"
    exec_time_column_name: str = "faithfulness_correlation_40.sample_exec_time"


class FaithfulnessCorrelation60PostProcessor(FaithfulnessCorrelation20PostProcessor):
    metric_name: str = "Faithfulness Correlation (60)"
    column_name: str = "faithfulness_correlation_60.faithfulness_corr_score"
    exec_time_column_name: str = "faithfulness_correlation_60.sample_exec_time"


class FaithfulnessEstimatePostProcessor(MetricPostProcessor):
    metric_name: str = "Faithfulness Estimate"
    column_name: str = "faithfulness_estimate.faithfulness_estimate_score"
    exec_time_column_name: str = "faithfulness_estimate.sample_exec_time"
    type: str = "faithfulness"
    metric_perturbation_type: str = "ordered_perturbation"
    is_lower_the_better: bool = False
    set_nan_zero: bool = True


class MonotonicityCorrelationPostProcessor(MetricPostProcessor):
    metric_name: str = "Monotonicity Correlation"
    column_name: str = "monotonicity_corr_and_non_sens.monotonicity_corr"
    exec_time_column_name: str = "monotonicity_corr_and_non_sens.sample_exec_time"
    type: str = "faithfulness"
    metric_perturbation_type: str = "unordered_perturbation"
    is_lower_the_better: bool = False
    set_nan_zero: bool = True


class InfidelityPostProcessor(MetricPostProcessor):
    metric_name: str = "Infidelity"
    column_name: str = "infidelity.infidelity_score"
    exec_time_column_name: str = "infidelity.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = True
    metric_perturbation_type: str = "unordered_perturbation"


class SensitivityN20PostProcessor(MetricPostProcessor):
    metric_name: str = "Sensitivity-N (20)"
    column_name: str = "sensitivity_n_20.sensitivity_n_score"
    exec_time_column_name: str = "sensitivity_n_20.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = True
    metric_perturbation_type: str = "unordered_perturbation"


class SensitivityN40PostProcessor(SensitivityN20PostProcessor):
    metric_name: str = "Sensitivity-N (40)"
    column_name: str = "sensitivity_n_40.sensitivity_n_score"
    exec_time_column_name: str = "sensitivity_n_40.sample_exec_time"


class SensitivityN60PostProcessor(SensitivityN20PostProcessor):
    metric_name: str = "Sensitivity-N (60)"
    column_name: str = "sensitivity_n_60.sensitivity_n_score"
    exec_time_column_name: str = "sensitivity_n_60.sample_exec_time"


class SufficiencyPostProcessor(MetricPostProcessor):
    metric_name: str = "Sufficiency"
    column_name: str = "aopc.baselines_perturbed_desc"
    exec_time_column_name: str = "aopc.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = True
    metric_perturbation_type: str = "ordered_perturbation"

    def _summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        target_len: int = 101,
        label: str | int | None = None,
    ) -> pd.Series:
        df = self.sanitize_df(df, is_multi_target=is_multi_target, label=label)
        if len(df) == 0:
            return {"score": np.nan}

        # take the original data
        baselines_perturbed_desc = df[self.column_name]

        # resample to the target length
        baselines_perturbed_desc = baselines_perturbed_desc.apply(
            resample_to_percentage, target_len=target_len
        )

        # take the mean across samples
        baselines_perturbed_desc = np.stack(baselines_perturbed_desc.to_numpy()).mean(0)

        # compute the mean AOPC value (last value in the curve)
        return {
            "score": baselines_perturbed_desc[-1],
        }


class ABPCPostProcessor(MetricPostProcessor):
    metric_name: str = "ABPC"
    column_name: List[str] = ["aopc.desc", "aopc.asc"]
    exec_time_column_name: str = "aopc.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = False
    metric_perturbation_type: str = "ordered_perturbation"

    def _summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,
        target_len: int = 101,
    ) -> pd.Series:
        df = self.sanitize_df(df, is_multi_target=is_multi_target, label=label)

        if len(df) == 0:
            return {
                "score": np.nan,
            }
        # take the original data
        aopc_desc = df[self.column_name[0]]
        aopc_asc = df[self.column_name[1]]

        # resample to the target length
        aopc_desc = aopc_desc.apply(resample_to_percentage, target_len=target_len)
        aopc_asc = aopc_asc.apply(resample_to_percentage, target_len=target_len)

        # take the mean across samples
        aopc_desc_mean = np.stack(aopc_desc.to_numpy()).mean(0)
        aopc_asc_mean = np.stack(aopc_asc.to_numpy()).mean(0)
        abpc_curve = aopc_desc_mean - aopc_asc_mean

        # plt.figure()
        # x = np.linspace(0, 100, target_len)
        # plt.plot(x, aopc_desc_mean, label="AOPC Descending", color="blue")
        # plt.plot(x, aopc_asc_mean, label="AOPC Ascending", color="orange")
        # plt.plot(x, abpc_curve, label="ABPC (Desc - Asc)", color="green")
        # plt.xlabel("Feature Percentage (%)")
        # plt.ylabel("AOPC Value")
        # plt.title("Mean AOPC Curves")
        # plt.legend()
        # plt.show()

        return {
            "score": abpc_curve[-1],
        }


class MonotonicityPostProcessor(MetricPostProcessor):
    metric_name: str = "Monotonicity"
    column_name: str = "monotonicity.monotonicity_score"
    exec_time_column_name: str = "monotonicity.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = False
    metric_perturbation_type: str = "ordered_perturbation"


class AOPCPostProcessor(MetricPostProcessor):
    metric_name: str = "AOPC"
    column_name: list[str] = ["aopc.desc", "aopc.asc", "aopc.rand"]
    exec_time_column_name: str = "aopc.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = False
    metric_perturbation_type: str = "ordered_perturbation"
    returns_single_value: bool = False

    def _summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,
        target_len: int = 101,
    ) -> pd.Series:
        # take the original data
        df = self.sanitize_df(df, is_multi_target=is_multi_target, label=label)

        if len(df) == 0:
            return {
                "aopc.desc": np.nan,
                "aopc.asc": np.nan,
                "aopc.rand": np.nan,
            }

        aopc_desc = df[self.column_name[0]]
        aopc_asc = df[self.column_name[1]]
        aopc_rand = df[self.column_name[2]].apply(lambda x: x.mean(0))

        # resample to the target length
        aopc_desc = aopc_desc.apply(resample_to_percentage, target_len=target_len)
        aopc_asc = aopc_asc.apply(resample_to_percentage, target_len=target_len)
        aopc_rand = aopc_rand.apply(resample_to_percentage, target_len=target_len)

        # take the mean across samples
        aopc_desc_mean = np.stack(aopc_desc.to_numpy()).mean(0)
        aopc_asc_mean = np.stack(aopc_asc.to_numpy()).mean(0)
        aopc_rand_mean = np.stack(aopc_rand.to_numpy()).mean(0)

        # plt.figure()
        # x = np.linspace(0, 100, target_len)
        # plt.plot(x, aopc_desc_mean, label="AOPC Descending", color="blue")
        # plt.plot(x, aopc_asc_mean, label="AOPC Ascending", color="orange")
        # plt.plot(x, aopc_rand_mean, label="AOPC Random", color="green")
        # plt.xlabel("Feature Percentage (%)")
        # plt.ylabel("AOPC Value")
        # plt.title("Mean AOPC Curves")
        # plt.legend()
        # plt.show()

        return {
            "aopc.desc": aopc_desc_mean.tolist(),
            "aopc.asc": aopc_asc_mean.tolist(),
            "aopc.rand": aopc_rand_mean.tolist(),
        }


class AOPCAscPostProcessor(MetricPostProcessor):
    metric_name: str = "AOPC (asc.)"
    column_name: str = "aopc.asc"
    exec_time_column_name: str = "aopc.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = True
    metric_perturbation_type: str = "ordered_perturbation"
    returns_single_value: bool = False


class AOPCRandPostProcessor(MetricPostProcessor):
    metric_name: str = "AOPC (rand.)"
    column_name: List[str] = [
        "aopc_inputs_fwd_batch",
        "aopc_inputs_perturbed_fwds_agg_batch",
    ]
    exec_time_column_name: str = "aopc.sample_exec_time"
    type: str = "faithfulness"
    is_lower_the_better: bool = False
    metric_perturbation_type: str = "ordered_perturbation"
    returns_single_value: bool = False


class SensitivityMaxPostProcessor(MetricPostProcessor):
    metric_name: str = "Sensitivity Max"
    column_name: str = "sensitivity_max_and_avg.sensitivity_max"
    exec_time_column_name: str = "sensitivity_max_and_avg.sample_exec_time"
    type: str = "robustness"
    is_lower_the_better: bool = True
    metric_perturbation_type: str | None = None


class SensitivityAvgPostProcessor(MetricPostProcessor):
    metric_name: str = "Sensitivity Avg."
    column_name: str = "sensitivity_max_and_avg.sensitivity_avg"
    exec_time_column_name: str = "sensitivity_max_and_avg.sample_exec_time"
    type: str = "robustness"
    is_lower_the_better: bool = True
    metric_perturbation_type: str | None = None


class ModalityTopkFraction(BaseModel):
    metric_name: str = "ModalityTopkFraction"
    column_name: str = "modality_topk_fraction"
    type: str = "diagnostics"

    def summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,  # NEW
    ) -> pd.Series:
        label_mask = None
        label_col = df["label"] if label is not None else None
        if label is not None and label_col is not None:
            if is_multi_target:
                label_mask = label_col.explode() == label
            else:
                label_mask = label_col == label

        modality_topk_values = df["diagnosis_metrics"].apply(
            lambda x: x.get("modality_topk_fraction", np.nan)
        )
        keys = modality_topk_values.iloc[0].keys()
        summarized = {}
        for key in keys:
            col = modality_topk_values.apply(
                lambda x: x.get(key, np.nan) if isinstance(x, dict) else np.nan
            )
            if is_multi_target:
                col = col.explode()

            col = col.apply(lambda x: x[0])
            if label_mask is not None:
                col = col[label_mask]
            summarized[key] = col.mean()

        if isinstance(summarized, np.float32):
            summarized = summarized.item()
        if isinstance(summarized, float):
            summarized = {"score": summarized}
        elif isinstance(summarized, pd.Series):
            summarized = summarized.to_dict()
        return {
            **summarized,
            "metric_perturbation_type": None,
            "type": self.type,
        }


class WordAttributionLocality(BaseModel):
    metric_name: str = "WordAttributionLocality"
    column_name: str = "word_attribution_locality"
    type: str = "diagnostics"

    def summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,  # NEW
    ) -> pd.Series:
        label_mask = None
        label_col = df["label"] if label is not None else None
        if label is not None and label_col is not None:
            if is_multi_target:
                label_mask = label_col.explode() == label
            else:
                label_mask = label_col == label

        word_attribution_locality_values = df["diagnosis_metrics"].apply(
            lambda x: x.get("word_attribution_locality", np.nan)
        )
        # if all nan values just return nan
        if word_attribution_locality_values.isna().all():
            return {
                "score": np.nan,
                "metric_perturbation_type": None,
                "type": self.type,
            }

        keys = word_attribution_locality_values.iloc[0].keys()
        summarized = {}
        for key in keys:
            col = word_attribution_locality_values.apply(
                lambda x: x.get(key, np.nan) if isinstance(x, dict) else np.nan
            )
            if is_multi_target:
                col = col.explode()

            col = col.apply(lambda x: x[0])
            if label_mask is not None:
                col = col[label_mask]
            summarized[key] = col.mean()

        if isinstance(summarized, np.float32):
            summarized = summarized.item()
        if isinstance(summarized, float):
            summarized = {"score": summarized}
        elif isinstance(summarized, pd.Series):
            summarized = summarized.to_dict()
        return {
            **summarized,
            "metric_perturbation_type": None,
            "type": self.type,
        }


def sanitize_text_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode a DataFrame where each row contains a list of token dicts
    (from diagnosis_metrics['text_profile']) into one row per token.

    Input row shape:
        df['diagnosis_metrics'] -> dict with key 'text_profile' -> list of lists of dicts
        df['label']             -> str or list[str] (for multi-target)

    Returns a flat DataFrame with columns:
        target_word, target_label, span_mean_gap, span_n_runs, ner_corr,
        ner_<class> (one col per NER class), row_idx (original row index)
    """
    records = []

    for row_idx, row in df.iterrows():
        profile_raw = row["diagnosis_metrics"].get("text_profile", [])
        # text_profile is list-of-lists; each inner list has one dict
        token_dicts = [item[0] for item in profile_raw if item]

        # label can be a scalar or list (multi-target)
        label = row.get("label", None)
        if isinstance(label, list):
            labels = label
        else:
            labels = [label]

        for token in token_dicts:
            ner_scores = token.get("ner", {})
            if ner_scores is None:
                ner_scores = {}
            rec = {
                "row_idx": row_idx,
                "labels": labels,  # keep as list for filtering
                "target_word": token.get("target_word"),
                "target_label": token.get("target_label"),
                "topk_mean_dist": token.get("topk_mean_dist"),
                "topk_max_dist": token.get("topk_max_dist"),
                "ner_corr": token.get("ner_corr"),
                "topk_words": token.get("topk_words", []),
                **{f"ner__{k}": v for k, v in ner_scores.items()},
            }
            records.append(rec)

    return pd.DataFrame(records)


def aggregate_text_profile(
    flat: pd.DataFrame,
    label: Optional[str] = None,
    is_multi_target: bool = False,
    top_n_words: int = 10,
) -> dict:
    """
    Aggregate a sanitized (flat) text-profile DataFrame.

    Parameters
    ----------
    flat            : output of sanitize_text_profile()
    label           : if given, restrict to tokens whose label set contains this value
    is_multi_target : if True, label column holds lists; filter uses .apply(... in ...)
    top_n_words     : how many top-k word frequencies to return

    Returns
    -------
    dict with keys:
        topk_word_freq  – Counter of words across all topk_words lists
        mean_ner        – dict[label_class -> float]
        mean_span_gap   – float
        mean_span_runs  – float
        mean_ner_corr   – float
    """
    # ── filter by label ───────────────────────────────────────────────────────
    if label is not None:
        if is_multi_target:
            mask = flat["labels"].apply(lambda lbls: label in lbls)
        else:
            mask = flat["labels"].apply(lambda lbls: lbls[0] == label)
        flat = flat[mask]

    if flat.empty:
        return {}

    # ── topk word frequencies ─────────────────────────────────────────────────
    word_counts: Counter = Counter()
    for words in flat["topk_words"]:
        word_counts.update(words)
    topk_freq = dict(word_counts.most_common(top_n_words))

    # ── mean NER mass per class ───────────────────────────────────────────────
    ner_cols = [c for c in flat.columns if c.startswith("ner__")]
    mean_ner = flat[ner_cols].mean().rename(lambda c: c[5:]).to_dict()  # strip 'ner__'

    return {
        "topk_word_freq": topk_freq,
        "mean_ner": mean_ner,
        "topk_mean_dist": float(flat["topk_mean_dist"].mean()),
        "topk_max_dist": float(flat["topk_max_dist"].mean()),
        "mean_ner_corr": float(flat["ner_corr"].mean()),
    }


class TextProfilePostProcessor(BaseModel):
    metric_name: str = "TextProfile"
    column_name: str = "text_profile"
    type: str = "diagnostics"
    top_n_words: int = 20

    def _get_flat(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and sanitize the flat text profile DataFrame from the raw df."""
        return sanitize_text_profile(df)

    def summarize(
        self,
        df: pd.DataFrame,
        is_multi_target: bool = False,
        label: str | int | None = None,
    ) -> dict:
        flat = self._get_flat(df)
        result = aggregate_text_profile(
            flat,
            label=label,
            is_multi_target=is_multi_target,
            top_n_words=self.top_n_words,
        )
        return {
            **result,
            "metric_perturbation_type": None,
            "type": self.type,
        }


# now we post-process the metrics per dataset
METRIC_POST_PROCESSORS: list[MetricPostProcessor] = [
    CompletenessPostProcessor(),
    ComplexityEntropyPostProcessor(),
    SundarajajanComplexityPostProcessor(),
    SparsenessPostProcessor(),
    EffectiveComplexityPostProcessor(),
    NonSensitivityPostProcessor(),
    FaithfulnessCorrelation20PostProcessor(),
    FaithfulnessCorrelation40PostProcessor(),
    FaithfulnessCorrelation60PostProcessor(),
    FaithfulnessEstimatePostProcessor(),
    MonotonicityCorrelationPostProcessor(),
    InfidelityPostProcessor(),
    SensitivityN20PostProcessor(),
    SensitivityN40PostProcessor(),
    SensitivityN60PostProcessor(),
    SufficiencyPostProcessor(),
    ABPCPostProcessor(),
    MonotonicityPostProcessor(),
    AOPCPostProcessor(),
    SensitivityMaxPostProcessor(),
    SensitivityAvgPostProcessor(),
    ModalityTopkFraction(),
    WordAttributionLocality(),
    TextProfilePostProcessor(),
]
