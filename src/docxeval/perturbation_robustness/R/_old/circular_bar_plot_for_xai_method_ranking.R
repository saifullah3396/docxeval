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
data <- read.csv(file.path(script_dir, output_dir, dataset, "dataset_wise_summarized_grouped_0_100.csv"))
# Replace type name by first symbol
data <- data %>% arrange(type, desc(score))

# se minmum score to 5
data$score <- ifelse(data$score < 5, 5, data$score)
data$score <- data$score * 0.5

# Set a number of 'empty bar' to add at the end of each type
data$type <- factor(data$type)
empty_bar <- 4
to_add <- data.frame(matrix(NA, empty_bar * nlevels(data$type), ncol(data)))
colnames(to_add) <- colnames(data)
to_add$type <- rep(levels(data$type), each = empty_bar)
print(to_add)
data <- rbind(data, to_add)
data <- data %>% arrange(type)
data$id <- seq(1, nrow(data))

# Get the name and the y position of each label
label_data <- data
number_of_bar <- nrow(label_data)
angle <- 90 - 360 * (label_data$id - 0.5) / number_of_bar # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
label_data$hjust <- ifelse(angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle + 180, angle)

# prepare a data frame for base lines
base_data <- data %>%
  group_by(type) %>%
  summarize(start = min(id), end = max(id) - empty_bar) %>%
  rowwise() %>%
  mutate(title = mean(c(start, end)))


# prepare a data frame for grid (scales)
grid_data <- base_data
grid_data$end <- grid_data$end[c(nrow(grid_data), 1:nrow(grid_data) - 1)] + 1
grid_data$start <- grid_data$start - 1
grid_data <- grid_data[-1, ]

# Make the plot
p <- ggplot(data, aes(x = as.factor(id), y = score, fill = type)) +
  geom_bar(aes(x = as.factor(id), y = score, fill = type), stat = "identity", alpha = 0.5) +

  # Add grid lines for reference
  geom_segment(data = grid_data, aes(x = end, y = 100 * 0.5, xend = start, yend = 100 * 0.5), colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE) +
  geom_segment(data = grid_data, aes(x = end, y = 80 * 0.5, xend = start, yend = 80 * 0.5), colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE) +
  geom_segment(data = grid_data, aes(x = end, y = 60 * 0.5, xend = start, yend = 60 * 0.5), colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE) +
  geom_segment(data = grid_data, aes(x = end, y = 40 * 0.5, xend = start, yend = 40 * 0.5), colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE) +
  geom_segment(data = grid_data, aes(x = end, y = 20 * 0.5, xend = start, yend = 20 * 0.5), colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE) +

  # Annotate score values
  annotate("text", x = rep(max(data$id), 5), y = c(20 * 0.5, 40 * 0.5, 60 * 0.5, 80 * 0.5, 100 * 0.5), label = c("20", "40", "60", "80", "100"), color = "grey", size = 3, angle = 0, fontface = "bold", hjust = 1) +

  # Barplot settings
  ylim(-50, 120) +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = c(0.8, 0.8), # Place legend at the top
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-1, 4), "cm"),
    legend.title = element_blank(), # Remove legend title
    legend.text = element_text(size = 10) # Style for legend items
  ) +
  coord_polar() +
  geom_text(
    data = label_data, aes(x = id, y = score + 10, label = XAI.Method, hjust = hjust),
    color = "black", fontface = "bold", alpha = 0.6, size = 2.5, angle = label_data$angle, inherit.aes = FALSE
  ) +

  # Add base line information
  geom_segment(
    data = base_data, aes(x = start, y = -5, xend = end, yend = -5),
    colour = "black", alpha = 0.5, size = 0.6, inherit.aes = FALSE
  ) +
  geom_text(
    data = base_data, aes(x = title, y = -18, label = substr(type, 1, 1)),
    hjust = c(1, 1, 0, 0), colour = "black", alpha = 0.8, size = 4, fontface = "bold", inherit.aes = FALSE
  ) +

  # Add color scale (use a predefined palette or customize colors)
  scale_fill_brewer(palette = "Set2", name = "Type")



# Save the plot to a file
print("Saving plot...")
output_path <- file.path(script_dir, output_dir, dataset, "xai_method_ranking.png")
# Save plot
ggsave(
  filename = output_path,
  plot = p,
  width = 6, # Set square dimensions
  height = 6,
  bg = "white",
  dpi = 300
)
