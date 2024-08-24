"""
Module Name: Plot Configuration for Seaborn Visualizations

Description:
This module sets up global plot configurations for Seaborn visualizations used across the project.
It defines font sizes, plot dimensions, legend settings, and a custom color palette to ensure
consistent and visually appealing plots.

Configurations:
- Font sizes for labels and ticks
- Plot dimensions
- Legend position and size
- Custom color palette
"""

import seaborn as sns

# Configuration for font sizes
X_LABEL_FONT_SIZE = 8  # Font size for the X-axis label
Y_LABEL_FONT_SIZE = 7  # Font size for the Y-axis label
X_TICK_FONT_SIZE = 8    # Font size for X-axis tick labels
Y_TICK_FONT_SIZE = 8    # Font size for Y-axis tick labels

# Configuration for plot dimensions (in inches)
PLOT_WIDTH = 4         # Width of the plot
PLOT_HEIGHT = 1.2      # Height of the plot

# Configuration for legend
LEGEND_SIZE = (1, 1)   # Dimensions of the legend box
LEGEND_POSITION = 'upper left'  # Position of the legend in the plot
LEGEND_FONT_SIZE = 6           # Font size for the legend text
LEGEND_TITLE_FONT_SIZE = 6     # Font size for the legend title

# Set Seaborn style and color palette
sns.set(style="whitegrid")  # Set Seaborn style to whitegrid for better readability
color_palette = sns.color_palette("deep")  # Use Seaborn's deep color palette

# Assign colors from the palette
primary_color = color_palette[0]
secondary_color = color_palette[1]
tertiary_color = color_palette[2]
quaternary_color = color_palette[3]

# List of colors for plots
colors = [primary_color, secondary_color, tertiary_color, quaternary_color]
