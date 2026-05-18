# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(hrbrthemes) # For clean and modern themes
library(RColorBrewer) # For color palettes
library(viridis) # For color palettes

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
output_dir <- ifelse(length(args) >= 1, args[1], "../../analysis_outputs")
dataset <- ifelse(length(args) >= 2, args[2], "Tobacco3482")

# Determine script directory
file <- commandArgs(trailingOnly = FALSE)[4]
file <- gsub("--file=", "", file)
script_dir <- dirname(normalizePath(file))

if (is.na(script_dir)) {
    script_dir <- dirname(sys.frame(1)$ofile)
}

# Load the data
aopc_desc <- read.csv(file.path(script_dir, output_dir, dataset, "dataset_wise_mean/AOPC (desc.).csv"))
aopc_asc <- read.csv(file.path(script_dir, output_dir, dataset, "dataset_wise_mean/AOPC (asc.).csv"))
aopc_rand <- read.csv(file.path(script_dir, output_dir, dataset, "dataset_wise_mean/AOPC (rand.).csv"))

# Combine data with an identifier for sets
aopc_desc <- aopc_desc %>% mutate(Set = "Set 1")
aopc_asc <- aopc_asc %>% mutate(Set = "Set 2")
aopc_rand <- aopc_rand %>% mutate(Set = "Set 3")

# Center random runs in Set 3, ignoring XAI.Method
aopc_rand_centered <- aopc_rand %>%
    group_by(aopc_x) %>%
    mutate(
        centered_random_run = random_run - mean(random_run, na.rm = TRUE),
        centered_aopc_y = aopc_y - mean(aopc_y, na.rm = TRUE)
    ) %>%
    ungroup()

# For Set 3: Calculate confidence intervals on centered random runs
set3_conf <- aopc_rand_centered %>%
    group_by(aopc_x) %>%
    summarise(
        mean_aopc_y = mean(centered_aopc_y, na.rm = TRUE),
        ymin = quantile(centered_aopc_y, 0.025, na.rm = TRUE),
        ymax = quantile(centered_aopc_y, 0.975, na.rm = TRUE),
        .groups = "drop"
    )

# Add a column to classify methods
gradient_methods <- c("Saliency", "DeepLift", "DeepLiftShap", "GradientShap", "InputXGradient", "IntegratedGradients") # Example gradient-based methods
perturbation_methods <- c("FeatureAblation", "Lime", "KernelShap", "Occlusion") # Example perturbation-based methods

aopc_desc <- aopc_desc %>%
    mutate(Method.Type = case_when(
        XAI.Method %in% gradient_methods ~ "Gradient-based",
        XAI.Method %in% perturbation_methods ~ "Perturbation-based",
        TRUE ~ "Other"
    ))

aopc_desc <- aopc_desc %>%
    mutate(XAI.Method = factor(XAI.Method, levels = c(
        gradient_methods,
        perturbation_methods,
        setdiff(unique(aopc_desc$XAI.Method), c(gradient_methods, perturbation_methods))
    )))

aopc_asc <- aopc_asc %>%
    mutate(Method.Type = case_when(
        XAI.Method %in% gradient_methods ~ "Gradient-based",
        XAI.Method %in% perturbation_methods ~ "Perturbation-based",
        TRUE ~ "Other"
    ))

# Rearrange the aopc_desc by Method.Type and XAI.Method to group the methods together
aopc_asc <- aopc_asc %>%
    mutate(XAI.Method = factor(XAI.Method, levels = c(
        gradient_methods,
        perturbation_methods,
        setdiff(unique(aopc_asc$XAI.Method), c(gradient_methods, perturbation_methods))
    )))

# Plot
p <- ggplot() +
    # Set 1: Normal lines
    geom_line(data = aopc_desc, aes(x = aopc_x, y = aopc_y, color = XAI.Method), size = 2, alpha = 0.65) +

    # # Set 2: Dashed lines
    geom_line(data = aopc_asc, aes(x = aopc_x, y = aopc_y, color = XAI.Method), size = 2, alpha = 0.65, linetype = "dashed") +

    # # Set 3: Confidence intervals with centered data
    geom_ribbon(data = set3_conf, aes(x = aopc_x, ymin = ymin, ymax = ymax), fill = "black", alpha = 0.4) +
    geom_line(data = set3_conf, aes(x = aopc_x, y = mean_aopc_y), color = "black", size = 1.25) +

    # Add theme and labels
    labs(
        x = "% Tokens Removed",
        y = "AOPC",
        color = "XAI Method",
        title = paste("AOPC Curves on ", dataset)
    ) +
    theme_gray(base_size = 18) + # Use a clean modern theme
    theme(
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold"),
        legend.spacing = unit(0.5, "cm"), # Add spacing between legend items
        plot.title = element_text(face = "bold", hjust = 0.5, margin = margin(b = 10)),
        axis.title = element_text(face = "bold"),
        plot.margin = margin(15, 15, 15, 15), # Ensure no clipping of legend
        aspect.ratio = 1
    ) +
    scale_color_viridis(discrete = TRUE) + # Use a color palette
    scale_color_viridis(discrete = TRUE) + # Use a color palette
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) # Split legend into 2 rows


# Save plot
ggsave(
    filename = file.path(script_dir, output_dir, paste0(dataset, "_aopc_plot.png")),
    plot = p,
    width = 14, # Set square dimensions
    height = 10,
    bg = "white"
)
