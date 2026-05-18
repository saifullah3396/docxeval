# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(hrbrthemes) # For clean and modern themes
library(RColorBrewer) # For color palettes
library(viridis)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
output_dir <- ifelse(length(args) >= 1, args[1], "../analysis_outputs")
dataset <- ifelse(length(args) >= 2, args[2], "Tobacco3482")

# Determine script directory
file <- commandArgs(trailingOnly = FALSE)[4]
file <- gsub("--file=", "", file)
script_dir <- dirname(normalizePath(file))
if (is.na(script_dir)) {
    script_dir <- dirname(sys.frame(1)$ofile)
}

# Load the data
data <- read.csv(file.path(script_dir, output_dir, dataset, "dataset_wise_mean/Random-Perturbation-Abs-Fwd-Diffs.csv"))

# Filter the dataframe for one XAI Method, for example, "DeepLift"
df_scores_filtered <- data %>%
    filter(XAI.Method == "DeepLift") # Replace "DeepLift" with your desired XAI method

# Calculate the total number of observations for each threshold separately
df_scores_filtered <- df_scores_filtered %>%
    group_by(threshold) %>%
    mutate(total_count = n())

# Define the symbol for threshold (gamma) and corresponding threshold values
df_scores_filtered$threshold_label <- paste("γ =", df_scores_filtered$threshold)

# Manually calculate the histogram data
binwidth <- 2
hist_data <- df_scores_filtered %>%
    group_by(threshold, percent_features_dropped) %>%
    mutate(score_bin = floor(score * 100 / binwidth) * binwidth) %>%
    count(threshold, score_bin) %>%
    mutate(percentage = n / sum(n) * 100) # Calculate percentages for each bin within the threshold

hist_data$threshold <- as.character(hist_data$threshold)

# Plot histogram for the 'score' column with percentages, with facetting by threshold
p <- ggplot(hist_data, aes(x = score_bin, y = percentage, fill = threshold, color = threshold)) +
    geom_bar(stat = "identity", position = "identity", alpha = 0.2, size = 0.5, width = binwidth) +
    # Labels and title
    labs(
        x = "% Perturbation Steps with Probability Drop less\nthan the Threshold (γ)",
        y = "Frequency of Observations",
        fill = "Threshold (γ)",
        color = "Threshold (γ)",
    ) +
    theme_gray(base_size = 18) + # Use a clean modern theme
    theme(
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold"),
        legend.spacing = unit(0.5, "cm"), # Add spacing between legend items
        plot.title = element_text(face = "bold", hjust = 0.5, margin = margin(b = 10)),
        axis.title = element_text(face = "bold"),
        plot.margin = margin(0, 0, 0, 0),
        aspect.ratio = 1
    ) +
    scale_color_manual(
        values = c("0.001" = "#FF6347", "0.01" = "#4682B4"), # Custom colors (replace with your desired colors)
    ) +
    scale_fill_manual(
        values = c("0.001" = "#FF6347", "0.01" = "#4682B4"), # Custom colors (replace with your desired colors)
    ) +
    ylim(0, 100) +
    facet_wrap(~percent_features_dropped, labeller = labeller(percent_features_dropped = function(x) paste("% of Feature Removed = ", as.numeric(x) * 10)))

# Save plot
ggsave(
    filename = file.path(script_dir, output_dir, paste0(dataset, "_random_feature_fwd_diff_hist.png")),
    plot = p,
    width = 15, # Set square dimensions
    height = 7.5,
    bg = "white",
    dpi = 300
)
