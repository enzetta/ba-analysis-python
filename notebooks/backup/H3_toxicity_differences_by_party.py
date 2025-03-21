# %% [markdown]
# # Hypothesis 3 Testing
# Testing toxicity differences between parties in terms of sent and received toxicity

# %%
# Data handling and analysis
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

# BigQuery
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

# Set up environment
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/zetta/projects/twitter-analysis-python/.secrets/service-account.json'

# Initialize BigQuery client
client = bigquery.Client()

# Output directory setup
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

# %% [markdown]
# ## 1. Data Collection
# Fetch toxicity data by party

# %%
# Query to get toxicity data by party
query = """
WITH party_mapping AS (
    SELECT party,
    CASE
        WHEN party = 'CDU' THEN 'cdu'
        WHEN party = 'SPD' THEN 'spdde'
        WHEN party = 'CSU' THEN 'csu'
        WHEN party = 'FDP' THEN 'fdp'
        WHEN party = 'Bündnis 90/Die Grünen' THEN 'die_gruenen'
        WHEN party = 'DIE LINKE' THEN 'dielinke'
        WHEN party = 'AfD' THEN 'afd'
    END AS party_id,
    CAST(avg_tox_sent AS FLOAT64) as avg_tox_sent,
    CAST(avg_tox_received AS FLOAT64) as avg_tox_received
FROM `grounded-nebula-408412.twitter_analysis_30_stats.toxcity_by_party`
WHERE party IN ('CDU', 'SPD', 'CSU', 'FDP', 'Bündnis 90/Die Grünen', 'DIE LINKE', 'AfD')
)
SELECT 
    party_id as party,
    avg_tox_sent,
    avg_tox_received
FROM party_mapping
WHERE avg_tox_sent IS NOT NULL
  AND avg_tox_received IS NOT NULL
"""

party_data = run_query(query)

# %% [markdown]
# ## 2. Kruskal-Wallis Test
# Testing whether there are significant differences in toxicity between parties

# %%
def perform_kruskal_wallis(df, metric):
    """Perform Kruskal-Wallis test for a given toxicity metric."""
    # Filter out NaN values
    df = df.dropna(subset=[metric])
    if len(df) == 0:
        return pd.DataFrame({
            'Metric': [metric],
            'H-statistic': [np.nan],
            'p-value': [np.nan],
            'Significant': [False]
        })
    
    groups = [group[metric].values for name, group in df.groupby('party')]
    if len(groups) < 2:
        return pd.DataFrame({
            'Metric': [metric],
            'H-statistic': [np.nan],
            'p-value': [np.nan],
            'Significant': [False]
        })
    
    h_stat, p_val = stats.kruskal(*groups)
    return pd.DataFrame({
        'Metric': [metric],
        'H-statistic': [h_stat],
        'p-value': [p_val],
        'Significant': [p_val < 0.05]
    })

# Run tests for toxicity metrics
metrics = ['avg_tox_sent', 'avg_tox_received']
kw_results = pd.concat([perform_kruskal_wallis(party_data, metric) for metric in metrics])

print("\nKruskal-Wallis Test Results:")
print(kw_results)

# Save results
timestamp = pd.Timestamp.now().strftime('%Y%m%d')
kw_results.to_csv(os.path.join(OUTPUT_DIR, f'h3_kruskal_wallis_{timestamp}.csv'), index=False)

# %% [markdown]
# ## 3. Mann-Whitney U Test
# Testing pairwise differences in toxicity between parties

# %%
def perform_mann_whitney_tests(df, metric):
    """
    Perform Mann-Whitney U tests between all pairs of parties for a given metric.
    """
    parties = df['party'].unique()
    results = []
    
    for i, party1 in enumerate(parties):
        for party2 in parties[i+1:]:
            group1 = df[df['party'] == party1][metric]
            group2 = df[df['party'] == party2][metric]
            
            # Perform Mann-Whitney U test
            u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Calculate effect size (r = Z / sqrt(N))
            n1, n2 = len(group1), len(group2)
            z_score = stats.norm.ppf(p_val / 2)
            effect_size = abs(z_score) / np.sqrt(n1 + n2)
            
            results.append({
                'Party 1': party1,
                'Party 2': party2,
                'Metric': metric,
                'U-statistic': u_stat,
                'p-value': p_val,
                'Effect Size': effect_size,
                'Significant': p_val < 0.05
            })
    
    return pd.DataFrame(results)

# Run Mann-Whitney U tests for both toxicity metrics
all_mw_results = []

for metric in metrics:
    mw_results = perform_mann_whitney_tests(party_data, metric)
    all_mw_results.append(mw_results)
    
    print(f"\nMann-Whitney U Test Results for {metric}:")
    significant_results = mw_results[mw_results['Significant']]
    if len(significant_results) > 0:
        print("\nSignificant differences found between:")
        for _, row in significant_results.iterrows():
            print(f"{row['Party 1']} vs {row['Party 2']}:")
            print(f"  U-statistic: {row['U-statistic']:.4f}")
            print(f"  p-value: {row['p-value']:.4e}")
            print(f"  Effect Size: {row['Effect Size']:.4f}")
    else:
        print("No significant differences found between any parties.")

# Save results
for i, metric in enumerate(metrics):
    results_df = all_mw_results[i]
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'h3_mann_whitney_{metric}_{timestamp}.csv'), index=False)

# Calculate descriptive statistics by party
desc_stats = party_data.groupby('party').agg({
    'avg_tox_sent': ['count', 'mean', 'median', 'std'],
    'avg_tox_received': ['count', 'mean', 'median', 'std']
}).round(4)

print("\nDescriptive Statistics by Party:")
print(desc_stats)

# %% [markdown]
# ## 4. Visualization of Mann-Whitney U Test Results

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def create_mann_whitney_heatmap(mw_results, metric, party_names):
    """Create a heatmap visualization for Mann-Whitney U test results."""
    # Get unique parties
    parties = sorted(set(mw_results['Party 1'].unique()) | set(mw_results['Party 2'].unique()))
    n_parties = len(parties)
    
    # Create matrices for heatmap and annotations
    heatmap_matrix = np.zeros((n_parties, n_parties))
    annot_matrix = np.empty((n_parties, n_parties), dtype=object)
    
    # Initialize matrices with empty strings
    for i in range(n_parties):
        for j in range(n_parties):
            annot_matrix[i, j] = ''
    
    # Fill matrices with values
    for _, row in mw_results.iterrows():
        i = parties.index(row['Party 1'])
        j = parties.index(row['Party 2'])
        
        # Format p-value with asterisks for significance
        sig_str = '***' if row['p-value'] < 0.001 else ('**' if row['p-value'] < 0.01 else ('*' if row['p-value'] < 0.05 else ''))
        
        # Format statistics
        if row['Significant']:
            stats_str = f"U={row['U-statistic']:.0f}\np={row['p-value']:.4f}{sig_str}\nr={row['Effect Size']:.2f}"
            
            # Store values in both positions of the matrix
            heatmap_matrix[i, j] = row['Effect Size']
            heatmap_matrix[j, i] = row['Effect Size']
            annot_matrix[i, j] = stats_str
            annot_matrix[j, i] = stats_str
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    ax = sns.heatmap(heatmap_matrix,
                     mask=mask,
                     xticklabels=[party_names.get(p, p) for p in parties],
                     yticklabels=[party_names.get(p, p) for p in parties],
                     cmap='YlOrRd',
                     center=0,
                     vmin=0,
                     vmax=max(mw_results['Effect Size']),
                     square=True,
                     cbar_kws={'label': 'Effect Size (r)'},
                     annot=annot_matrix,
                     fmt='',
                     annot_kws={'color': 'white', 'fontsize': 11})
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Set title
    plt.title(f'Effect Sizes of Significant Differences in {metric}', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(OUTPUT_DIR, f'mann_whitney_heatmap_{metric}_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# Create heatmaps for both metrics
PARTY_NAMES = {
    "cdu": "CDU",
    "spdde": "SPD",
    "csu": "CSU",
    "fdp": "FDP",
    "die_gruenen": "Die Grünen",
    "dielinke": "Die Linke",
    "afd": "AfD"
}

for i, metric in enumerate(metrics):
    create_mann_whitney_heatmap(all_mw_results[i], metric, PARTY_NAMES)

# %%
