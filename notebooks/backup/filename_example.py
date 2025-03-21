import os
import matplotlib.pyplot as plt
from utils import generate_unique_filename

# Example of how to use the unique filename generator in your notebook
# Replace your current code:
# file_name = 'avg_toxicity_by_party_affiliation.png'
# plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=PLOT_DPI, bbox_inches='tight')

# With this code:
file_name = generate_unique_filename('avg_toxicity_by_party_affiliation.png')
# This will create a filename like: avg_toxicity_by_party_affiliation_20230415_123045.png
plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=PLOT_DPI, bbox_inches='tight')

# You can also add additional information to the filename if needed:
# For example, to include the data source or parameters:
base_name = 'toxicity_by_topic'
parameters = 'climate_migration'
file_name = generate_unique_filename(f"{base_name}_{parameters}.png")
# This will create a filename like: toxicity_by_topic_climate_migration_20230415_123045.png 