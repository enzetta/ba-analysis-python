import json
import time
import os

from enum import Enum
from pydantic import BaseModel, Field
from typing import List

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from google.cloud import bigquery

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Create a logger instance
logger = logging.getLogger(__name__)


# Configuration
ENTRY_LIMIT = 10000
TARGET_FILE = "accounts_classified.csv"

PRICE_PER_1M_INPUT_TOKENS = {
    "gpt-4o-mini": 0.15,  # $0.150 per 1M input tokens
    "gpt-4o": 2.50,  # $2.50 per 1M input tokens
    "gpt-4o-2024-11-20": 2.50,  # $2.50 per 1M input tokens
    "gpt-4": 30.00,  # $30.00 per 1M input tokens
}

PRICE_PER_1M_OUTPUT_TOKENS = {
    "gpt-4o-mini": 0.60,  # $0.600 per 1M output tokens
    "gpt-4o": 10.00,  # $10.00 per 1M output tokens
    "gpt-4o-2024-11-20": 10.00,  # $10.00 per 1M output tokens
    "gpt-4": 60.00,  # $60.00 per 1M output tokens
}

MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
    "gpt-4": "gpt-4o",
}
MODEL = MODELS["gpt-4o"]

# Retry configuration
MAX_RETRIES = 5
MIN_WAIT_SECONDS = 1
MAX_WAIT_SECONDS = 60

# BigQuery configuration
PROJECT_ID = "grounded-nebula-408412"
DATASET = "twitter_analysis_30_network_analysis"

TARGET_DATASET = "base_twitter_python"
TARGET_TABLE = "account_types"


# BigQuery query to get top accounts by pagerank
QUERY = f"""
SELECT
  node_id,
  party
FROM
  `grounded-nebula-408412.twitter_analysis_30_network_analysis.nodes_all`
WHERE
  node_id IN (
  SELECT
    node_id
  FROM
    `grounded-nebula-408412.twitter_analysis_30_network_analysis.nodes_all`
  WHERE
    month_start = "2021-09-01"
    AND node_id NOT IN ("fcbayern", "youtube", "tracklist", "actufoot_", "fcbayernen", "miele", "fcunion")
  ORDER BY
    pagerank DESC
  LIMIT
    {ENTRY_LIMIT} )
  AND month_start = "2021-09-01"
ORDER BY
  pagerank DESC
"""


class AccountTypeEnum(str, Enum):
    """Enumeration of all Account Types"""
    MEDIA_NEWS = "Media/News"
    POLITICIAN = "Politician"
    POLITICAL_PARTY = "Political Party"
    INSTITUTION = "Institution"
    BUSINESS = "Business"
    ACTIVIST_GROUP = "Activist Group"
    OTHER = "Other"

class AccountClassification(BaseModel):
    """Model for account classification response"""
    account_type: AccountTypeEnum = Field(..., description="The classified account type")
    confidence: float = Field(..., description="Confidence score between 0 and 1")


def run_bigquery(query):
    """Execute a BigQuery query and return results as a DataFrame."""
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=PROJECT_ID)

        # Execute the query
        query_job = client.query(query)

        # Convert to DataFrame
        results_df = query_job.to_dataframe()

        logger.info(f"Query executed successfully. Retrieved {len(results_df)} rows.")

        return results_df

    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        # Return an empty DataFrame so code can continue without errors
        return pd.DataFrame()

def write_to_bigquery(df, dataset, table):
    """Write DataFrame to BigQuery, creating or replacing the table."""
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=PROJECT_ID)

        # Define table reference
        table_ref = f"{PROJECT_ID}.{dataset}.{table}"

        # Define job configuration
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Replace if exists
            schema=[
                bigquery.SchemaField("node_id", "STRING", description="Twitter account node ID"),
                bigquery.SchemaField("party", "STRING", description="Political party affiliation if available"),
                bigquery.SchemaField("account_type", "STRING", description="Classified account type"),
                bigquery.SchemaField("confidence", "FLOAT", description="Confidence score of classification")
            ]
        )

        # Load data to BigQuery
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for job to complete

        # Get the table
        table = client.get_table(table_ref)

        logger.info(f"Loaded {table.num_rows} rows into {table_ref}")
        return True

    except Exception as e:
        logger.error(f"Error writing to BigQuery: {str(e)}")
        return False


# Retry decorator using tenacity
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=MIN_WAIT_SECONDS, max=MAX_WAIT_SECONDS),
    before_sleep=lambda retry_state: logger.info(
        f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
    ),
)
def make_api_request(client, node_id, party, model):
    """Make an API request with retry logic"""
    return client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a Twitter account classifier and super expert in the german political and media landscape. Classify the Twitter account type based on the node_id and party information.",
            },
            {
                "role": "user",
                "content": f"""Classify Twitter account with node_id `{node_id}` and party `{party if party else 'Unknown'}` into one of these ENUM categories.
                """,
            },
        ],
        response_format=AccountClassification,

        temperature=0.0,
    )


def classify_accounts():
    """Main function to fetch accounts from BigQuery and classify them"""

    # Get accounts from BigQuery
    logger.info("Fetching accounts from BigQuery...")
    accounts_df = run_bigquery(QUERY)

    if accounts_df.empty:
        logger.info("No accounts retrieved from BigQuery. Exiting.")
        return

    # Limit the number of accounts if needed
    if ENTRY_LIMIT and len(accounts_df) > ENTRY_LIMIT:
        accounts_df = accounts_df.head(ENTRY_LIMIT)

    logger.info(f"Processing {len(accounts_df)} accounts...")

    # Initialize OpenAI client
    openai_client = OpenAI()

    # Initialize tracking variables
    classifications = []
    total_input_tokens = 0
    total_output_tokens = 0

    # Process each account
    for index, row in accounts_df.iterrows():
        node_id = row["node_id"]
        party = row["party"] if "party" in row and not pd.isna(
            row["party"]) else None

        logger.info(
            f"Processing {index + 1}/{len(accounts_df)}: @{node_id} (Party: {party if party else 'Unknown'})\n"
        )

        try:
            # Make API request with retry logic
            response = make_api_request(openai_client, node_id, party, MODEL)

            # Parse response
            try:
                result = json.loads(response.choices[0].message.content)
                account_type = result.get("account_type", "Other")
                confidence = result.get("confidence", 0.0)
            except Exception as parse_error:
                logger.error(f"Error parsing response: {str(parse_error)}")
                account_type = "Other"
                confidence = 0.0

            # Track tokens
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

            # Store results
            classifications.append({
                "node_id": node_id,
                "party": party,
                "account_type": account_type,
                "confidence": confidence
            })

            # Print progress
            logger.info(f"  Account Type: {account_type}")
            logger.info(f"  Confidence: {confidence:.2f}")

        except Exception as e:
            logger.error(
                f"Failed to process account '{node_id}' after all retries: {str(e)}"
            )
            # Store failure result
            classifications.append({
                "node_id": node_id,
                "party": party,
                "account_type": "Other",
                "confidence": 0.0
            })

        time.sleep(0.1)  # Small delay to avoid rate limits

    # Create DataFrame with classifications
    classifications_df = pd.DataFrame(classifications)

    # Merge with original data (we keep all columns from accounts_df)
    result = pd.merge(accounts_df,
                      classifications_df,
                      on=["node_id", "party"],
                      how="left")

    # Save results
    # result.to_csv(TARGET_FILE, index=False)

    # Save results to BigQuery
    logger.info("Writing results to BigQuery...")
    bigquery_success = write_to_bigquery(classifications_df, TARGET_DATASET, TARGET_TABLE)


    # Print summary
    logger.info("\nClassification Complete!")
    logger.info(f"Total Input Tokens: {total_input_tokens:,}")
    logger.info(f"Total Output Tokens: {total_output_tokens:,}")

    # Calculate cost
    # input_cost = (total_input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS[MODEL.split("-")[0]]
    # output_cost = (total_output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS[MODEL.split("-")[0]]
    # total_cost = input_cost + output_cost

    # print(f"Estimated Cost: ${total_cost:.4f}")
    if bigquery_success:
        logger.info(f"Results saved to BigQuery table: {PROJECT_ID}.{TARGET_DATASET}.{TARGET_TABLE}")
    else:
        logger.error("Failed to save results to BigQuery table.")

if __name__ == "__main__":
    classify_accounts()
