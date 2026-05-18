# Load required libraries
library(ggplot2)
library(dplyr)

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
output_dir <- ifelse(length(args) >= 1, args[1], "../../analysis_outputs")
dataset <- ifelse(length(args) >= 2, args[2], "Tobacco3482")

# Determine script directory
script_dir <- dirname(normalizePath(commandArgs(trailingOnly = FALSE)[4]))

# Determine script directory
file <- commandArgs(trailingOnly = FALSE)[4]
file <- gsub("--file=", "", file)
script_dir <- dirname(normalizePath(file))

if (is.na(script_dir)) {
    script_dir <- dirname(sys.frame(1)$ofile)
}

# Build file path
data <- read.csv(file.path(script_dir, output_dir, dataset, "dataset_wise_summarized_xai_metrics.csv"))

# Add a column to classify methods
gradient_methods <- c("Saliency", "DeepLift", "DeepLiftShap", "GradientShap", "InputXGradient", "IntegratedGradients") # Example gradient-based methods
perturbation_methods <- c("FeatureAblation", "Lime", "KernelShap", "Occlusion") # Example perturbation-based methods

data <- data %>%
    mutate(Method.Type = case_when(
        XAI.Method %in% gradient_methods ~ "Gradient-based",
        XAI.Method %in% perturbation_methods ~ "Perturbation-based",
        TRUE ~ "Other"
    ))

# Rearrange the data by Method.Type and XAI.Method to group the methods together
data <- data %>%
    mutate(XAI.Method = factor(XAI.Method, levels = c(
        gradient_methods,
        perturbation_methods,
        setdiff(unique(data$XAI.Method), c(gradient_methods, perturbation_methods))
    )))


# Create the plot with visual enhancements
p <- ggplot(data, aes(x = XAI.Method, y = score, fill = XAI.Method)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    theme_minimal(base_size = 14) + # Use a clean minimal theme
    theme(
        axis.title.x = element_text(size = 16, face = "bold", family = "Arial"), # Bold title with larger font
        axis.title.y = element_text(size = 16, face = "bold", family = "Arial"),
        axis.text.x = element_text(size = 14, angle = 90, hjust = 1, family = "Arial"), # Larger axis text with rotation
        axis.text.y = element_text(size = 14, family = "Arial"),
        legend.title = element_text(size = 14, face = "bold", family = "Arial"), # Bold legend title
        legend.text = element_text(size = 12, family = "Arial"),
        legend.position = "none", # Remove the legend as it's redundant with x-axis
        strip.text = element_text(size = 14, face = "bold", family = "Arial"), # Facet label formatting
        panel.grid.major = element_line(size = 0.1, color = "grey80"), # Subtle grid lines
        panel.grid.minor = element_blank(), # Minor grid lines removed
        plot.title = element_text(size = 18, face = "bold", family = "Arial", hjust = 0.5), # Centered, bold title
    ) +
    geom_text(aes(label = sprintf("%.2f", score)),
        vjust = 0, color = "black",
        position = position_dodge(0.8), size = 4, angle = 0, hjust = 0.5
    ) +
    labs(
        x = "XAI Method",
        y = "Score",
        title = "Evaluation Metrics by XAI Method"
    ) +
    scale_fill_brewer(palette = "Set3") + # Use color-blind friendly palette
    facet_wrap(~ type + formatted_name, scales = "free_y") + # Facet by Evaluation Metric
    theme(strip.background = element_rect(fill = "lightgray", color = NA, size = 0.5))

# Save the plot with the updated aspect ratio
ggsave(
    file.path(script_dir, output_dir, dataset, "xai_methods_comparison.png"),
    plot = p, width = 20, height = 15, dpi = 300, bg = "white"
)
