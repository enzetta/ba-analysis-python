import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
import seaborn as sns

# Define style constants (matching the notebook style)
TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
LEGEND_POSITION = 'upper right'
LEGEND_FRAME_ALPHA = 0.9
LINE_WIDTH = 2

# Define colors
colors = ['#4A4A4A',  # Dark Gray - Good for "Overall" or neutral data
          '#4C72B0',  # Muted Blue - Calm, reliable
          '#C44E52',  # Muted Red - Strong, noticeable
          '#55A868',  # Muted Green - Natural, balanced
          '#8172B2',  # Muted Purple - Soft but distinct
          '#CCB974']  # Muted Yellow - Warm, highlight-friendly
sns.set_palette(sns.color_palette(colors))

# Load the network metrics data
network_metrics = pd.read_csv('data/output/2025-02-13_10-05-08_network_users_metrics_latest.csv')

# Convert month_start to datetime
network_metrics['month_start'] = pd.to_datetime(network_metrics['month_start'])

# Sort by date
network_metrics = network_metrics.sort_values('month_start')

# Create a line chart showing modularity and toxicity over time
fig, ax1 = plt.subplots(figsize=(12, 6))

# Calculate the maximum value for both metrics to set consistent y-axis limits
max_modularity = network_metrics['modularity'].max()
max_toxicity = network_metrics['network_avg_toxicity'].max()
y_max = max(max_modularity, max_toxicity) * 1.1  # Add 10% padding

# Plot modularity on the left y-axis
ax1.set_xlabel('Month', fontsize=AXIS_LABEL_SIZE)
ax1.set_ylabel('Modularity', fontsize=AXIS_LABEL_SIZE, color=colors[1])
ax1.plot(network_metrics['month_start'], network_metrics['modularity'], 
         marker='o', markersize=4, linewidth=LINE_WIDTH, color=colors[1], label='Modularity')
ax1.tick_params(axis='y', labelcolor=colors[1])
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, y_max)  # Set y-axis to start at 0 and end at the common max value

# Create a second y-axis for toxicity
ax2 = ax1.twinx()
ax2.set_ylabel('Network Average Toxicity', fontsize=AXIS_LABEL_SIZE, color=colors[2])
ax2.plot(network_metrics['month_start'], network_metrics['network_avg_toxicity'], 
         marker='o', markersize=4, linewidth=LINE_WIDTH, color=colors[2], label='Network Avg Toxicity')
ax2.tick_params(axis='y', labelcolor=colors[2])
ax2.set_ylim(0, y_max)  # Set y-axis to start at 0 and end at the common max value

# Add title
plt.title('Network Modularity and Toxicity Over Time', fontsize=TITLE_SIZE, pad=20)

# Format x-axis dates
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

# Add legend - moved to the right side
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
           frameon=True, framealpha=LEGEND_FRAME_ALPHA, 
           edgecolor='gray', fontsize=LEGEND_FONT_SIZE)

# Add summary statistics as text
modularity_stats = (
    f"Modularity statistics:\n"
    f"Mean: {network_metrics['modularity'].mean():.3f}\n"
    f"Median: {network_metrics['modularity'].median():.3f}\n"
    f"Min: {network_metrics['modularity'].min():.3f}\n"
    f"Max: {network_metrics['modularity'].max():.3f}"
)

toxicity_stats = (
    f"Toxicity statistics:\n"
    f"Mean: {network_metrics['network_avg_toxicity'].mean():.3f}\n"
    f"Median: {network_metrics['network_avg_toxicity'].median():.3f}\n"
    f"Min: {network_metrics['network_avg_toxicity'].min():.3f}\n"
    f"Max: {network_metrics['network_avg_toxicity'].max():.3f}"
)

# Add text boxes with statistics - both on the right side
# Modularity stats below the legend
ax1.text(0.98, 0.80, modularity_stats, transform=ax1.transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=LEGEND_FRAME_ALPHA, 
                  edgecolor=colors[1], pad=0.5))

# Toxicity stats below the modularity stats
ax1.text(0.98, 0.55, toxicity_stats, transform=ax1.transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=LEGEND_FRAME_ALPHA,
                  edgecolor=colors[2], pad=0.5))

plt.tight_layout()
plt.savefig('notebooks/outputs/modularity_toxicity_time_series.png', dpi=300)
plt.show()

# Create a second visualization showing the relationship between modularity and toxicity
plt.figure(figsize=(10, 6))

# Create a scatter plot with connected lines
scatter = plt.scatter(network_metrics['modularity'], network_metrics['network_avg_toxicity'], 
                     c=network_metrics['month_start'].apply(lambda x: x.toordinal()), 
                     cmap='viridis', s=80, zorder=5)

# Connect points with lines in chronological order
plt.plot(network_metrics['modularity'], network_metrics['network_avg_toxicity'], 
         '-', color='gray', alpha=0.5, linewidth=1, zorder=1)

# Add a colorbar to show the time progression
cbar = plt.colorbar(scatter)
cbar.set_label('Time', fontsize=AXIS_LABEL_SIZE)
# Format the colorbar ticks to show dates
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.ax.set_yticklabels([pd.to_datetime(ordinal, origin='julian').strftime('%b %Y') 
                         for ordinal in cbar.get_ticks()])

# Add labels and title
plt.xlabel('Modularity', fontsize=AXIS_LABEL_SIZE)
plt.ylabel('Network Average Toxicity', fontsize=AXIS_LABEL_SIZE)
plt.title('Relationship Between Network Modularity and Toxicity', fontsize=TITLE_SIZE, pad=20)
plt.grid(True, linestyle='--', alpha=0.7)

# Calculate and add correlation information
correlation = network_metrics['modularity'].corr(network_metrics['network_avg_toxicity'])
corr_text = f"Correlation: {correlation:.3f}"
plt.text(0.02, 0.02, corr_text, transform=plt.gca().transAxes, 
         verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=LEGEND_FRAME_ALPHA))

plt.tight_layout()
plt.savefig('notebooks/outputs/modularity_toxicity_relationship.png', dpi=300)
plt.show() 