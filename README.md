# Twitter Analysis for Political Communication
## Bachelor Thesis Project - Humboldt-Universität zu Berlin

## Overview
This project analyzes Twitter data to understand political communication patterns, focusing on sentiment analysis, toxicity levels, and network relationships between political actors. The analysis is designed to help understand how different political parties and actors communicate on social media during various electoral phases.

Parts of this project's codebase were developed with assistance from Large Language Models (LLMs), specifically OpenAI's GPT models and Anthropic's Claude. Additionally, the project actively uses OpenAI's GPT-4 API for two specific classification tasks:
- Classification of Twitter accounts into categories (politicians, media, institutions)
- Classification of hashtags into predefined political topics

## Research Goals
- Analyze sentiment and toxicity in political tweets
- Examine network relationships between political actors
- Study how communication patterns change during electoral phases
- Investigate differences in communication styles between political parties

## Project Structure
The project is organized into several key directories, each serving a specific purpose:

### Directory Overview
```bash
twitter-analysis-python/
├── notebooks/                    # Hypothesis testing and result visualization
│   ├── overview.ipynb            # Main analysis workflow
│   ├── node_metrics.ipynb       # Network metrics evaluation
│   ├── H1_network_correlations.ipynb
│   ├── H2A_network_kruskal_wallis_electoral_phases.ipynb
│   ├── H3_toxicity_differences_by_party.ipynb
│   ├── outputs/                 # Generated visualizations and results
│   ├── outputs_BACKUP/          # Backup of analysis outputs
│   └── backup/                  # Notebook backups
├── toxicity/                    # Tweet toxicity classification
│   ├── toxicity.py             # Core toxicity classification
│   ├── predict_v3.py           # Batch classification pipeline
│   ├── evaluate.py             # Classification evaluation metrics
│   ├── bigquery_client.py      # BigQuery data operations
│   ├── hyperparameter_tester.py # Model optimization
│   └── backup/                  # Backup implementations
├── network/                     # Network metrics calculation
│   ├── calculate_network_metrics.py # Network-level metrics
│   ├── calculate_node_metrics.py    # Node-level metrics
│   └── analyse_network.py           # Network structure analysis
├── analysis/                    # Data visualization utilities
│   ├── visualise.py            # Visualization functions
│   └── recipe.py               # Data processing recipes
├── account_classification/      # Account type classification
│   └── classify_account_type.py # Political account classifier
├── hashtags/                    # Hashtag classification
│   ├── classify_hashtags.py     # Topic classification
│   ├── format_for_bigquery.py   # Data formatting
│   └── get_hashtag_counts.sql   # Hashtag frequency query
├── .secrets/                    # Credential storage
│   └── service-account.json     # Google Cloud credentials
├── accounts_classified.csv      # Classified account results
├── bigquery_schema.json        # Database schema definition
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

### `/notebooks` - Hypothesis Testing and Visualization
Contains Jupyter notebooks for testing research hypotheses and visualizing results:
- `main.ipynb` - Initial analysis pipeline and data processing
- `node_metrics.ipynb` - Evaluation of network centrality and community metrics
- `H1_network_correlations.ipynb` - Tests hypotheses about network relationship patterns
- `H2A_network_kruskal_wallis_electoral_phases.ipynb` - Statistical analysis of temporal communication patterns
- `H3_toxicity_differences_by_party.ipynb` - Analysis of toxicity variations across political parties
- `/outputs` - Generated visualizations and statistical results

### `/toxicity` - Tweet Toxicity Classification
Multi-model classification pipeline using transformer-based models:
- `toxicity.py` - Implementation using XLM-RoBERTa for toxicity and German BERT for sentiment
- `predict_v3.py` - Batch processing pipeline for large-scale tweet classification
- `evaluate.py` - Model performance evaluation and metric calculation
- `bigquery_client.py` - Database Client for reading and storing data to BigQuery
- `hyperparameter_tester.py` - Model optimization and performance tuning

### `/network` - Network Metrics Calculation
Network analysis implementation using NetworkX:
- `calculate_network_metrics.py` - Computes global network metrics (density, modularity)
- `calculate_node_metrics.py` - Calculates node-level metrics (centrality, influence)
- `analyse_network.py` - Community detection and network structure analysis

### `/analysis` - Visualization Utilities
Standardized visualization and data processing:
- `visualise.py` - Matplotlib/Seaborn visualization functions
- `recipe.py` - Common data transformation and processing recipes

### `/account_classification` - Account Type Classification
GPT-4 based classification script for Twitter accounts:
- `classify_account_type.py` - Classifies accounts into categories (Media/News, Politician, Political Party, Institution, etc.)
- Uses OpenAI's API with retry mechanisms and confidence scoring

### `/hashtags` - Hashtag Classification
LLM-powered hashtag topic classification:
- `classify_hashtags.py` - Categorizes hashtags into 50+ predefined topics using GPT models
- `format_for_bigquery.py` - Formats classification results for database storage
- `get_hashtag_counts.sql` - Retrieves hashtag frequency statistics

## Technical Requirements

### Development Environment
- Python 3.11.3
- Google Cloud SDK
- Virtual environment manager (venv)

### Environment Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Required Environment Variables:**
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_APPLICATION_CREDENTIALS=.secrets/service-account.json
   ```
   See `.env.example` for reference.

4. **Google Cloud Configuration:**
   - Store the Google Cloud service account key in `.secrets/service-account.json`
   - Ensure the service account has appropriate permissions for BigQuery read/write access

## Data Processing Pipeline

1. **Data Collection**
   - Tweet data storage in Google BigQuery
   - Raw data schema includes tweet content, timestamps, and user information

2. **Analysis Process**
   - Sentiment and toxicity analysis using machine learning models
   - Network analysis of political relationships
   - Hashtag categorization and trend analysis
   - Statistical testing of hypotheses

3. **Results Storage**
   - Analysis results are stored in BigQuery
   - Visualizations are saved in the `/notebooks/outputs` directory
   - Raw data processing results in CSV format

## Running Analyses

### 1. Sentiment Analysis
```bash
python toxicity/predict_v3.py
```
This processes tweets and analyzes their sentiment and toxicity levels.

### 2. Network Analysis
Use the notebooks in the `/notebooks` directory:
1. Open Jupyter Notebook or Jupyter Lab
2. Navigate to the relevant hypothesis testing notebook
3. Run cells sequentially to reproduce analysis

### 3. Generating Visualizations
The notebooks automatically generate visualizations in `/notebooks/outputs`, including:
- Network graphs
- Toxicity trends over time
- Party comparison charts
- Statistical test results

## Notes for Evaluators
This project demonstrates:
- Implementation of state-of-the-art natural language processing models
- Automated and reproducible data processing pipelines
- Rigorous statistical methods documented in notebooks
- Professional data management using Google BigQuery
- Robust error handling and logging mechanisms

## Technical Support
For questions regarding the technical implementation:
1. Refer to the inline documentation in Python files
2. Consult the methodology sections in Jupyter notebooks
3. Contact the thesis author through official university channels

## Academic Context
This project was developed as part of a bachelor thesis at Humboldt-Universität zu Berlin and is subject to university guidelines for academic work.
