from typing import Optional

import numpy as np
import scipy
import torch


def compute_non_sens(
    perturbed_fwd_diffs_relative_vars: np.ndarray,
    feature_group_attribution_scores: np.ndarray,
    zero_attribution_threshold: float = 1.0e-03,
    zero_variance_threshold: float = 1.0e-01,
    return_ratio: bool = False,
    use_percentage_attribution_threshold: bool = True,
):
    n_features = feature_group_attribution_scores.shape[0]

    # find the indices of features that have a zero attribution score, every attribution score value less
    # than non_sens_eps is considered zero
    def find_small_scale_features(
        values: np.ndarray, threshold: Optional[float] = None
    ):
        return set(list(np.argwhere(np.abs(values) < threshold).flatten()))

    if np.sum(feature_group_attribution_scores) == 0:
        zero_attribution_features = set(range(n_features))
    else:
        if use_percentage_attribution_threshold:
            feature_group_attribution_scores = (
                feature_group_attribution_scores
                / np.sum(feature_group_attribution_scores)
            )

        # find the indices of features that have a zero attribution score.
        zero_attribution_features = find_small_scale_features(
            feature_group_attribution_scores,
            threshold=zero_attribution_threshold,
        )

    # find the indices of features that have a zero model forward variance,
    # all values below the threshold are considered zero. Default threshold is set to 1%
    zero_variance_features = find_small_scale_features(
        perturbed_fwd_diffs_relative_vars,
        threshold=zero_variance_threshold,
    )

    # import matplotlib.pyplot as plt

    # # Plot zero attribution features
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.bar(
    #     range(n_features),
    #     feature_group_attribution_scores,
    #     color="blue",
    #     alpha=0.6,
    #     label="Attribution Scores",
    # )
    # plt.scatter(
    #     list(zero_attribution_features),
    #     feature_group_attribution_scores[list(zero_attribution_features)],
    #     color="red",
    #     label="Zero Attribution Features",
    # )
    # plt.xlabel("Feature Index")
    # plt.ylabel("Attribution Score")
    # plt.title("Zero Attribution Features")
    # plt.legend()

    # # Plot zero variance features
    # plt.subplot(1, 2, 2)
    # plt.bar(
    #     range(n_features),
    #     perturbed_fwd_diffs_relative_vars,
    #     color="green",
    #     alpha=0.6,
    #     label="Forward Variances",
    # )
    # plt.scatter(
    #     list(zero_variance_features),
    #     perturbed_fwd_diffs_relative_vars[list(zero_variance_features)],
    #     color="red",
    #     label="Zero Variance Features",
    # )
    # plt.xlabel("Feature Index")
    # plt.ylabel("Forward Variance")
    # plt.title("Zero Variance Features")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # find the symmetric difference of the zero attribution features and the zero variance features
    # this set should be empty if the model is non-sensitive to the zero attribution features
    # symmetric difference will give the oppposite of the intersection of the two sets
    # therefore non-sensitivity corresponds to the number of features that have either:
    # 1. zero attribution scores and non-zero model forward variances
    # 2. non-zero attribution scores and zero model forward variances
    # a higher non-sensitivity score indicates that the model is more sensitive to the zero attribution features
    # and a lower non-sensitivity score indicates that the model is non-sensitive to the zero attribution features
    non_sens = len(
        zero_attribution_features.symmetric_difference(zero_variance_features)
    )
    if return_ratio:
        return non_sens / n_features
    return non_sens


# compute effective complexity metric
def compute_effective_complexity(
    perturbed_fwd_diffs_relative_vars: torch.Tensor | np.ndarray,
    zero_variance_threshold: float = 1.0e-01,
    return_ratio: bool = True,
):
    # if the variance in output it less than a threshold, that means the feature set is not important
    # find top-k features that are important
    # this implementation assumes that the perturbed_fwd_diffs_relative_vars only increases as the features
    # are removed. It could be that it goes up and down, but it is not clear how that should be handled?
    # should we find the first first drop in the variance and stop there?

    # import matplotlib.pyplot as plt

    # # Fit isotonic regression
    # iso_reg = IsotonicRegression(increasing=True)
    # indices = np.arange(len(perturbed_fwd_diffs_relative_vars))
    # y_iso = iso_reg.fit_transform(indices, perturbed_fwd_diffs_relative_vars)

    # # Calculate the differences in the fitted curve
    # iso_differences = np.diff(y_iso)
    # # Find the point of maximum increase
    # max_increase_idx = np.argmax(iso_differences)
    # print("max_increase_idx", max_increase_idx)
    # plt.plot(y_iso)
    # plt.show()
    if not isinstance(perturbed_fwd_diffs_relative_vars, torch.Tensor):
        assert isinstance(perturbed_fwd_diffs_relative_vars, np.ndarray)
        perturbed_fwd_diffs_relative_vars = torch.from_numpy(
            perturbed_fwd_diffs_relative_vars
        )

    N = len(perturbed_fwd_diffs_relative_vars)

    # original implementation takes count of all values > threshold but that is incorrect
    # we need to find the first index where the threshold is crossed
    top_k_indices = torch.where(
        perturbed_fwd_diffs_relative_vars > zero_variance_threshold
    )[0]
    if len(top_k_indices) > 0:
        top_k_features = N - top_k_indices[0].item()
    else:
        top_k_features = 0
    # plt.axvline(
    #     x=first_index,
    #     color="r",
    #     linestyle="--",
    #     label=f"first_significant_index {top_k_features} features",
    # )
    # plt.plot(perturbed_fwd_diffs_relative_vars)
    # plt.show()
    if return_ratio:
        return top_k_features / N
    else:
        return top_k_features


# compute monotonocity corr metric
def compute_monotonocity_corr(
    perturbed_fwd_diffs_relative_vars: np.ndarray,
    feature_group_attribution_scores: np.ndarray,
):
    # find the spearman corr between the attribution scores and the model forward variances
    # this corr should be close to 1 if the model forward variances are monotonically increasing with the attribution scores
    # this means that features that have a lower attribution score are directly correlated with lower effect on the model output
    return scipy.stats.spearmanr(
        feature_group_attribution_scores,
        perturbed_fwd_diffs_relative_vars,
    )[0]


def compute_abpc(aopc_scores: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        morf = (
            aopc_scores[:, 0, :].numpy().mean(0)
        )  # first row is descending, take mean over the dataset
        lerf = (
            aopc_scores[:, 1, :].numpy().mean(0)
        )  # second row is ascending, take mean over the dataset
        abpc = morf - lerf
        return abpc


def compute_abpc_scores(
    morf_scores: torch.Tensor, lerf_scores: torch.Tensor
) -> torch.Tensor:
    return compute_abpc(
        torch.stack(
            [torch.from_numpy(morf_scores), torch.from_numpy(lerf_scores)], dim=1
        )
    )


def compute_selectivity(descending_perturbation_fwds: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        from scipy.integrate import simpson

        return simpson(
            descending_perturbation_fwds,
            x=np.arange(0, descending_perturbation_fwds.shape[0])
            / (descending_perturbation_fwds.shape[0] - 1),
        )


def compute_aopc_scores_vectorized(
    inputs_perturbed_fwds: torch.Tensor, input_fwds: torch.Tensor
) -> torch.Tensor:
    """
    Computes the AOPC score in a vectorized manner for the given input perturbations and forward outputs
    for a single sample.

    Args:
        inputs_perturbed_fwds (torch.Tensor): The forward outputs of the model on the perturbed inputs. The shape of
            the tensor is [(2*n_random_perms), n_features]. The first row corresponds to the descending order of feature
            importance, the second row corresponds to the ascending order of feature importance and the rest of the rows
            correspond to the random order of feature importance.
        input_fwds (torch.Tensor): The forward output of the model on the original input.
    """

    # concatenate the input forward output with the perturbed input forward outputs
    input_fwds = input_fwds.unsqueeze(-1)
    cat_fwds = torch.cat(
        [input_fwds.repeat(inputs_perturbed_fwds.shape[0], 1), inputs_perturbed_fwds],
        dim=1,
    )

    # Convert input_fwds to tensor for broadcasting
    input_fwds_tensor = input_fwds.expand_as(cat_fwds)

    # Compute the differences between input_fwds and each score
    differences = input_fwds_tensor - cat_fwds

    # Compute cumulative sum along the rows (axis=1)
    cumulative_sums = torch.cumsum(differences, dim=1)

    # Compute the number of elements considered so far
    counts = torch.arange(
        1, inputs_perturbed_fwds.shape[1] + 2, device=cumulative_sums.device
    ).float()
    counts = counts.expand_as(cumulative_sums)

    # Compute AOPC scores
    aopc_scores = cumulative_sums / counts

    return aopc_scores
