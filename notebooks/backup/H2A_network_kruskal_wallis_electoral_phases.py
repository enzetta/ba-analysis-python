# %% [markdown]
# # Network Toxicity Analysis Over Time
# 
# This notebook analyzes toxicity across different network types and time periods.

# %%
# Data handling and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

# BigQuery
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Set up environment
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/zetta/projects/twitter-analysis-python/.secrets/service-account.json'

# Create BigQuery client
client = bigquery.Client()

# Output directory setup
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Dataset configurations - only keep the 'all' network
DATASETS = {
    'all': {
        'table': 'grounded-nebula-408412.twitter_analysis_00_source_python.network_metrics_all',
        'name': 'All Topics Network',
        'output_prefix': 'all'
    }
}

# Plotting configurations
PLOT_FIGURE_SIZE = (12, 12 / 16 * 9)
PLOT_DPI = 300
BASE_FONT_SIZE = 14

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = PLOT_FIGURE_SIZE
plt.rcParams['font.size'] = BASE_FONT_SIZE

# %%
def run_query(query, use_cache=True):
    """Execute a BigQuery query and return results as a DataFrame."""
    try:
        job_config = bigquery.QueryJobConfig(use_query_cache=use_cache)
        query_job = client.query(query, job_config=job_config)
        results_df = query_job.to_dataframe(create_bqstorage_client=False)
        print(f"Query executed successfully. Retrieved {len(results_df)} rows.")
        return results_df
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return pd.DataFrame()

def format_plot(ax, title=None, xlabel=None, ylabel=None, ylim_start=0, ylim_end=None):
    """Apply standard formatting to a matplotlib axis."""
    if title:
        ax.set_title(title, fontweight='regular', pad=15, fontsize=BASE_FONT_SIZE + 2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=BASE_FONT_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE)
    if ylim_end is not None:
        ax.set_ylim(ylim_start, ylim_end)
    elif ylim_start > 0:
        ax.set_ylim(bottom=ylim_start)
    
    ax.grid(True, linestyle="--", alpha=0.5, color="#E0E0E0")
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def fetch_network_data():
    """Fetch toxicity data from the overall network."""
    query = """
    SELECT 
        month_start,
        network_avg_toxicity
    FROM `grounded-nebula-408412.twitter_analysis_00_source_python.network_metrics_all`
    ORDER BY month_start
    """
    
    df = run_query(query)
    if df.empty:
        return pd.DataFrame()
    
    df['month_start'] = pd.to_datetime(df['month_start'])
    df['year'] = df['month_start'].dt.year
    df['quarter'] = df['month_start'].dt.quarter
    df['month'] = df['month_start'].dt.month
    
    # Add electoral period
    def get_electoral_period(date):
        if date < pd.Timestamp('2020-10-01'):
            return 'Non-Electoral'
        elif date < pd.Timestamp('2021-04-01'):
            return 'Pre-Campaign'
        elif date < pd.Timestamp('2021-07-01'):
            return 'Campaign'
        elif date < pd.Timestamp('2021-09-01'):
            return 'Intensive Campaign'
        elif date < pd.Timestamp('2021-10-01'):
            return 'Final Sprint'
        else:
            return 'Post-Election'
    
    df['electoral_period'] = df['month_start'].apply(get_electoral_period)
    
    # Define the order of electoral periods for plotting
    df['electoral_period'] = pd.Categorical(
        df['electoral_period'],
        categories=['Non-Electoral', 'Pre-Campaign', 'Campaign', 
                   'Intensive Campaign', 'Final Sprint', 'Post-Election'],
        ordered=True
    )
    
    return df

def perform_temporal_kruskal_wallis(df):
    """Perform Kruskal-Wallis tests for toxicity across different time periods."""
    # By Electoral Period
    electoral_groups = [group['network_avg_toxicity'].values 
                       for name, group in df.groupby('electoral_period')]
    h_stat, p_val = stats.kruskal(*electoral_groups)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'h_statistic': [h_stat],
        'p_value': [p_val]
    }, index=['Electoral Periods'])
    
    return results

def save_results_to_csv(results_dict, filename):
    """Save statistical results to CSV file."""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d')
    for name, df in results_dict.items():
        output_path = os.path.join(OUTPUT_DIR, f'toxicity_network_kw_test_{name}_{timestamp}.csv')
        df.to_csv(output_path)
        print(f"Saved results to {output_path}")

def plot_toxicity_trends(df):
    """Create visualizations for toxicity trends."""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d')
    
    # Time series plot with electoral periods
    plt.figure(figsize=(15, 8))
    plt.plot(df['month_start'], df['network_avg_toxicity'], 
            marker='o', color='blue', linewidth=2, markersize=6)
    
    # Add electoral period backgrounds
    periods = df['electoral_period'].unique()
    period_colors = plt.cm.Pastel1(np.linspace(0, 1, len(periods)))
    
    for period, color in zip(periods, period_colors):
        period_data = df[df['electoral_period'] == period]
        plt.axvspan(period_data['month_start'].min(), period_data['month_start'].max(),
                   alpha=0.2, color=color, label=period)
    
    plt.title('Network Average Toxicity Over Time', pad=20, fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Average Toxicity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Electoral Period', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot and data
    plt.savefig(os.path.join(OUTPUT_DIR, f'overall_network_toxicity_time_series_{timestamp}.png'), 
                dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # Box plot by electoral period
    plt.figure(figsize=(15, 8))
    ax = sns.boxplot(data=df, x='electoral_period', y='network_avg_toxicity', 
                    color='skyblue')
    
    plt.title('Distribution of Network Toxicity by Electoral Period', pad=20, fontsize=14)
    plt.xlabel('Electoral Period', fontsize=12)
    plt.ylabel('Average Toxicity', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add mean line
    means = df.groupby('electoral_period')['network_avg_toxicity'].mean()
    for i, period in enumerate(df['electoral_period'].unique()):
        plt.plot([i-0.4, i+0.4], 
                [means[period], means[period]], 
                color='red', linestyle='--', alpha=0.8)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save plot and data
    plt.savefig(os.path.join(OUTPUT_DIR, f'overall_network_toxicity_by_period_{timestamp}.png'), 
                dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # Save statistical summaries
    period_stats = df.groupby('electoral_period')['network_avg_toxicity'].describe()
    period_stats.to_csv(os.path.join(OUTPUT_DIR, f'overall_network_toxicity_stats_{timestamp}.csv'))
    
    # Save complete dataset
    df.to_csv(os.path.join(OUTPUT_DIR, f'overall_network_toxicity_data_{timestamp}.csv'), index=False)

# %% [markdown]
# ## Load and Prepare Data

# %%
# Fetch data from the overall network
network_data = fetch_network_data()
print("Data shape:", network_data.shape)
print("\nSample of the data:")
print(network_data.head())

# %% [markdown]
# ## Perform Kruskal-Wallis Test

# %%
# Run Kruskal-Wallis test across electoral periods
kw_results = perform_temporal_kruskal_wallis(network_data)
print("\nKruskal-Wallis Test Results:")
print(kw_results)

# Save results
kw_results.to_csv(os.path.join(OUTPUT_DIR, 
                              f'overall_network_toxicity_kw_test_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'))

# %% [markdown]
# ## Perform Mann-Whitney U-tests

# %%
# Run Mann-Whitney U-tests between all pairs of electoral periods
mw_results = perform_mann_whitney_tests(network_data)

# Sort results by p-value
mw_results_sorted = mw_results.sort_values('p-value')

# Save results
timestamp = pd.Timestamp.now().strftime('%Y%m%d')
mw_results_sorted.to_csv(os.path.join(OUTPUT_DIR, f'overall_network_toxicity_mann_whitney_{timestamp}.csv'), index=False)

# Display results
print("\nMann-Whitney U-test Results (sorted by p-value):")
pd.set_option('display.max_rows', None)
print(mw_results_sorted)

# Create summary visualization of significant differences
plt.figure(figsize=(12, 8))
significant_pairs = mw_results[mw_results['Significant']]

# Create heatmap matrix
periods = network_data['electoral_period'].unique()
n_periods = len(periods)
heatmap_matrix = np.zeros((n_periods, n_periods))
np.fill_diagonal(heatmap_matrix, -1)  # Diagonal will be gray

# Create annotation matrix
annot_matrix = np.empty((n_periods, n_periods), dtype=object)
for i in range(n_periods):
    for j in range(n_periods):
        annot_matrix[i, j] = ''

# Fill matrices with values
for _, row in mw_results.iterrows():
    i = np.where(periods == row['Period 1'])[0][0]
    j = np.where(periods == row['Period 2'])[0][0]
    
    # Format p-value with asterisks for significance
    sig_str = '***' if row['p-value'] < 0.001 else ('**' if row['p-value'] < 0.01 else ('*' if row['p-value'] < 0.05 else ''))
    
    # Format test statistic and effect size
    value_str = f"U={row['U-statistic']:.0f}\np={row['p-value']:.3f}{sig_str}\nr={row['Effect Size']:.2f}"
    
    # Store values in both positions of the matrix
    annot_matrix[i, j] = value_str
    annot_matrix[j, i] = value_str
    
    if row['Significant']:
        heatmap_matrix[i, j] = row['Effect Size']
        heatmap_matrix[j, i] = row['Effect Size']

# Create mask for upper triangle and diagonal
mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))

# Plot heatmap
sns.heatmap(heatmap_matrix, 
            mask=mask,
            xticklabels=periods,
            yticklabels=periods,
            cmap='YlOrRd',
            center=0,
            vmin=0,
            vmax=max(mw_results['Effect Size']),
            square=True,
            cbar_kws={'label': 'Effect Size'},
            annot=annot_matrix,
            fmt='',
            annot_kws={'color': 'white', 'fontsize': 11})

plt.title('Significant Differences in Toxicity Between Electoral Periods\n(Effect Size of Significant Pairs)', 
          pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add significance legend
plt.figtext(1.02, 0.02, '* p<0.05\n** p<0.01\n*** p<0.001', 
            fontsize=10, ha='left')

# Save plot with extra right margin for legend
plt.savefig(os.path.join(OUTPUT_DIR, f'overall_network_toxicity_mann_whitney_heatmap_{timestamp}.png'), 
            dpi=PLOT_DPI, bbox_inches='tight', 
            pad_inches=0.5)
plt.close()

# %% [markdown]
# ## Visualize Toxicity Trends

# %%
# Create visualization plots and save data
plot_toxicity_trends(network_data) 