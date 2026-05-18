library(ggplot2)
library(hrbrthemes)

# -----------------------------
# Parse command-line arguments
# -----------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript script.R <input_csv> <output_png>")
}

input_csv  <- args[1]
output_png <- args[2]

cat("Input CSV:", input_csv, "\n")
cat("Output PNG:", output_png, "\n")

# -----------------------------
# Load data
# -----------------------------
data <- read.csv(input_csv, stringsAsFactors = FALSE)
data$percent_features_perturbed <- as.character(data$percent_features_perturbed)

# -----------------------------
# Plot
# -----------------------------
p <- ggplot(
  data,
  aes(
    x = percent_features_perturbed,
    y = perf,
    fill = strategy_type,
    color = strategy_type
  )
) +
  geom_boxplot(width = 0.5, alpha = 0.8, outlier.shape = NA) +
  facet_wrap(~ modality, scales = "free") +
  labs(
    y = "Change in Performance (compared to baseline)",
    x = "% Features Removed",
    title = "Modality-Wise Model Sensitivity to Different Feature Removal Strategies"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  ) +
  scale_fill_discrete(name = "Removal Strategy") +
  scale_color_discrete(name = "Removal Strategy")

# -----------------------------
# Save plot
# -----------------------------
cat("Saving plot...\n")
ggsave(
  filename = output_png,
  plot = p,
  width = 10,
  height = 7
)