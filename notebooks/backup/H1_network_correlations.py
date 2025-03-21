# %% [markdown]
# # Network Metrics Correlation Analysis
# 
# This notebook analyzes correlations between various network metrics from our Twitter dataset.

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
from matplotlib.colors import LinearSegmentedColormap

# Set up environment
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/zetta/projects/twitter-analysis-python/.secrets/service-account.json'

# Create BigQuery client
client = bigquery.Client()

# Output directory setup
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Dataset configurations
DATASETS = {
    'all': {
        'table': 'grounded-nebula-408412.twitter_analysis_00_source_python.network_metrics_all',
        'name': 'All Topics Network',
        'output_prefix': 'all'
    },
    'climate': {
        'table': 'grounded-nebula-408412.twitter_analysis_00_source_python.network_metrics_climate',
        'name': 'Climate Network',
        'output_prefix': 'climate'
    },
    'migration': {
        'table': 'grounded-nebula-408412.twitter_analysis_00_source_python.network_metrics_migration',
        'name': 'Migration Network',
        'output_prefix': 'migration'
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

# Selected metrics for analysis
SELECTED_METRICS = [
    "modularity",
    "network_avg_toxicity",
    "transitivity",
    "assortativity",
    "max_core_number",
    "rich_club_coefficient",
    "average_clustering",
    "connected_components",
    "density",
]

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

def normalize_series(series):
    """Normalize a series using z-score normalization."""
    return (series - series.mean()) / series.std()

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

def calculate_statistical_measures(df, metrics, normalize=True):
    """Calculate various statistical measures between metrics."""
    results = {}
    
    for i, metric1 in enumerate(metrics):
        results[metric1] = {}
        for metric2 in metrics[i + 1:]:
            x = df[metric1].values
            y = df[metric2].values
            
            if normalize:
                x = normalize_series(pd.Series(x))
                y = normalize_series(pd.Series(y))
            
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            y_pred = p(x)
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            
            results[metric1][metric2] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'r_squared': r_squared,
                'slope': z[0],
                'intercept': z[1]
            }
    
    return results

def calculate_lagged_correlations(df, metrics, max_lag=3, normalize=True):
    """Calculate lagged correlations between selected metrics."""
    lag_correlations = {}
    
    for metric1 in metrics:
        lag_correlations[metric1] = {}
        for metric2 in metrics:
            if metric1 != metric2:
                lag_correlations[metric1][metric2] = []
                for lag in range(max_lag + 1):
                    if lag == 0:
                        x = df[metric1].values
                        y = df[metric2].values
                    else:
                        x = df[metric1][lag:].values
                        y = df[metric2][:-lag].values
                    
                    if normalize:
                        x = normalize_series(pd.Series(x))
                        y = normalize_series(pd.Series(y))
                    
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    y_pred = p(x)
                    r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
                    corr = stats.spearmanr(x, y)[0]
                    
                    lag_correlations[metric1][metric2].append({
                        'lag': lag,
                        'correlation': corr,
                        'r_squared': r_squared
                    })
    
    return lag_correlations

def run_query_for_dataset(dataset_config):
    """Execute a BigQuery query for a specific dataset and return results as a DataFrame."""
    query = f"""
    SELECT 
        month_start,
        nodes,
        edges,
        density,
        connected_components,
        transitivity,
        modularity,
        modularity_classes,
        assortativity,
        network_avg_toxicity,
        median_node_toxicity,
        max_core_number,
        avg_core_number,
        rich_club_coefficient,
        average_clustering
    FROM `{dataset_config['table']}`
    ORDER BY month_start
    """
    
    try:
        job_config = bigquery.QueryJobConfig(use_query_cache=True)
        query_job = client.query(query, job_config=job_config)
        results_df = query_job.to_dataframe(create_bqstorage_client=False)
        print(f"Query executed successfully for {dataset_config['name']}. Retrieved {len(results_df)} rows.")
        return results_df
    except Exception as e:
        print(f"Error executing query for {dataset_config['name']}: {str(e)}")
        return pd.DataFrame()

# %% [markdown]
# ## Fetch Network Metrics Data
# 
# Let's retrieve network metrics from all three datasets for correlation analysis.

# %%
# Fetch data for all datasets
network_metrics_dfs = {}
for dataset_key, dataset_config in DATASETS.items():
    network_metrics_dfs[dataset_key] = run_query_for_dataset(dataset_config)

# %% [markdown]
# ## Correlation Analysis
# 
# Let's analyze the correlations between different network metrics using correlation matrices and heatmaps for each dataset.

# %%
for dataset_key, df in network_metrics_dfs.items():
    dataset_config = DATASETS[dataset_key]
    print(f"\nAnalyzing {dataset_config['name']}:")
    
    # Calculate statistical measures
    stats_results = calculate_statistical_measures(df, SELECTED_METRICS)
    lag_correlations = calculate_lagged_correlations(df, SELECTED_METRICS)

    # Calculate correlations for selected metrics
    selected_df = df[SELECTED_METRICS]
    correlation_matrix = selected_df.corr(method='spearman')

    # Create correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu', 
                center=0,
                vmin=-1,
                vmax=1,
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8})

    plt.title(f'Correlation Matrix of Network Metrics - {dataset_config["name"]}', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_config["output_prefix"]}_network_metrics_correlation.png'), 
                dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()

    # Create R² matrix
    r_squared_matrix = pd.DataFrame(index=SELECTED_METRICS, columns=SELECTED_METRICS)

    # Fill the matrix
    for i, metric1 in enumerate(SELECTED_METRICS):
        for j, metric2 in enumerate(SELECTED_METRICS):
            if metric1 == metric2:
                r_squared_matrix.loc[metric1, metric2] = 1.0
            else:
                if metric1 in stats_results and metric2 in stats_results[metric1]:
                    r_squared = stats_results[metric1][metric2]['r_squared']
                elif metric2 in stats_results and metric1 in stats_results[metric2]:
                    r_squared = stats_results[metric2][metric1]['r_squared']
                else:
                    x = normalize_series(df[metric1])
                    y = normalize_series(df[metric2])
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    y_pred = p(x)
                    r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
                
                r_squared_matrix.loc[metric1, metric2] = r_squared
                r_squared_matrix.loc[metric2, metric1] = r_squared

    r_squared_matrix = r_squared_matrix.astype(float)

    # Create R² heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(r_squared_matrix, 
                annot=True, 
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8})

    plt.title(f'R² Matrix of Network Metrics - {dataset_config["name"]}', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_config["output_prefix"]}_network_metrics_r_squared.png'), 
                dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()

    # Time series analysis
    plt.figure(figsize=(15, 8))
    for metric in SELECTED_METRICS:
        normalized = normalize_series(df[metric])
        plt.plot(df['month_start'], normalized, label=metric, linewidth=2)

    plt.title(f'Normalized Metric Trends Over Time - {dataset_config["name"]}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_config["output_prefix"]}_normalized_trends.png'), 
                dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()

    # Save detailed statistical measures
    output_file = os.path.join(OUTPUT_DIR, f'{dataset_config["output_prefix"]}_detailed_statistical_measures.txt')
    with open(output_file, 'w') as f:
        f.write(f"Detailed Statistical Measures for {dataset_config['name']}:\n")
        print(f"\nDetailed Statistical Measures for {dataset_config['name']}:")
        
        for metric1 in stats_results:
            for metric2 in stats_results[metric1]:
                header = f"\n{metric1} vs {metric2}:"
                f.write(header + "\n")
                print(header)
                
                for measure, value in stats_results[metric1][metric2].items():
                    line = f"  {measure}: {value:.4f}"
                    f.write(line + "\n")
                    print(line)
                
                max_lag = max(lag_correlations[metric1][metric2], 
                             key=lambda x: abs(x['correlation']))
                lag_line = (f"  Strongest lag: {max_lag['lag']} months "
                           f"(correlation: {max_lag['correlation']:.4f}, "
                           f"R²: {max_lag['r_squared']:.4f})")
                f.write(lag_line + "\n")
                print(lag_line)

    print(f"\nDetailed statistical measures for {dataset_config['name']} have been saved to: {output_file}")

# %% [markdown]
# ## Toxicity Correlations Analysis
# 
# Let's analyze how network metrics correlate with toxicity across all three networks.

# %%
for dataset_key, df in network_metrics_dfs.items():
    dataset_config = DATASETS[dataset_key]
    print(f"\nAnalyzing toxicity correlations for {dataset_config['name']}:")
    
    stats_results = calculate_statistical_measures(df, SELECTED_METRICS)
    toxicity_metric = "network_avg_toxicity"
    other_metrics = [m for m in SELECTED_METRICS if m != toxicity_metric]

    for metric in other_metrics:
        if toxicity_metric in stats_results and metric in stats_results[toxicity_metric]:
            stats_result = stats_results[toxicity_metric][metric]
        elif metric in stats_results and toxicity_metric in stats_results[metric]:
            stats_result = stats_results[metric][toxicity_metric]
        
        print(f"\n{metric}:")
        print(f"  Spearman r: {stats_result['spearman_r']:.3f} (p = {stats_result['spearman_p']:.3e})")
        print(f"  R²: {stats_result['r_squared']:.3f}")
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=PLOT_FIGURE_SIZE)
        
        x = normalize_series(df[toxicity_metric])
        y = normalize_series(df[metric])
        
        ax.scatter(x, y, alpha=0.5, color='#4C72B0')
        
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), '--', color='#C44E52', alpha=0.8)
        
        stats_text = (
            f"Pearson correlation:\n"
            f"  r = {stats_result['pearson_r']:.3f}\n"
            f"  p = {stats_result['pearson_p']:.3e}\n"
            f"\nSpearman correlation:\n"
            f"  r = {stats_result['spearman_r']:.3f}\n"
            f"  p = {stats_result['spearman_p']:.3e}\n"
            f"\nRegression:\n"
            f"  R² = {stats_result['r_squared']:.3f}\n"
            f"  slope = {stats_result['slope']:.3f}\n"
            f"  intercept = {stats_result['intercept']:.3f}"
        )
        
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        format_plot(ax,
                    title=f'{toxicity_metric} vs {metric} - {dataset_config["name"]}',
                    xlabel=f'{toxicity_metric} (normalized)',
                    ylabel=f'{metric} (normalized)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_config["output_prefix"]}_{toxicity_metric}_{metric}_comparison.png'), 
                    dpi=PLOT_DPI, bbox_inches='tight')
        plt.show() 