import pandas as pd
from datetime import datetime
import logging
import os
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize BigQuery client
client = bigquery.Client()

dataset = "twitter_analysis_curated"
project_id = "grounded-nebula-408412"

# Output configuration
OUTPUT_DIR = os.path.join("data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

networks = [
    {
        "name": "general",
        "table_name": ""
    },
    {
        "name": "migration",
        "table_name": ""
    },
    {
        "name": "climate",
        "table_name": ""
    },
]

metrics = ["pagerank", "betweeness", "..."]


# histogram - distribution
def query_metric(metric):
    query_template = f"""
    SELECT *
    {metric}
    FROM {project_id}.{dataset}.{table_name}
    WHERE month_start = @month_start

    """


save_csv()
save_visualisation()
# ...
