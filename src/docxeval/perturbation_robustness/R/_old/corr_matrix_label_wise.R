# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(hrbrthemes) # For clean and modern themes
library(RColorBrewer) # For color palettes

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
data <- read.csv(file.path(script_dir, output_dir, dataset, "label_wise_corr.csv"))

p <- ggplot(data, aes(Label.X, Label.Y, fill = Correlation)) +
    geom_tile() +
    facet_wrap(~type) +
    # theme_gray(base_size = 18) + # Use a clean modern theme
    theme(
        # legend.position = "bottom",
        panel.background = element_rect(fill = "white", color = NA),
        # legend.box = "horizontal",
        # legend.title = element_text(face = "bold"),
        # legend.spacing = unit(0.5, "cm"), # Add spacing between legend items
        plot.title = element_blank(),
        axis.title = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1),
        # plot.margin = margin(0, 0, 0, 0), # Ensure no clipping of legend
        axis.ticks = element_blank(),
        aspect.ratio = 1
    ) +
    scale_fill_distiller(palette = "Blues")

# Save plot
ggsave(
    filename = file.path(script_dir, output_dir, dataset, "label_wise_corr.png"),
    plot = p,
    width = 10, # Set square dimensions
    height = 10,
    bg = "white",
    dpi = 300
)
