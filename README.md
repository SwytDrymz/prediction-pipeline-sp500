# S&P 500 Prediction Pipeline

An automated data pipeline that fetches S&P 500 market data, calculates technical indicators, and generates daily price movement predictions using machine learning models.

## Features

- **Automated Data Collection**: Fetches daily market data for all S&P 500 components from Yahoo Finance (via `finfetcher`).
- **Technical Analysis**: Calculates indicators like RSI, MACD, Bollinger Bands, ATR, and more using `pandas-ta`.
- **Machine Learning**: Implements a classification model (Decision Tree) to predict daily returns.
- **Database Integration**: Stores historical market data, predictions, and model evaluations in a PostgreSQL database.
- **Daily Execution**: Configured with GitHub Actions to run every day at 00:00 UTC.
- **Evaluation Loop**: Automatically evaluates previous predictions as new market data becomes available.

## Project Structure

```text
├── .github/workflows/   # CI/CD configuration
├── src/
│   ├── api/             # API clients (if any)
│   ├── models/          # ML model definitions (Base & Specialized)
│   ├── pipeline/        # Core pipeline logic (Collector, Runner, Database)
│   └── utils/           # Shared utilities (Logging, DB initialization)
├── main.py              # Entry point for the daily pipeline
└── requirements.txt     # Project dependencies
```

## Prerequisites

- Python 3.10+
- PostgreSQL database
- (Optional) GitHub account for automated daily runs

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sp500_pipeline.git
cd sp500_pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Create a `.env` file in the root directory with your PostgreSQL connection string:

```text
DATABASE_URL=postgresql://user:password@localhost:5432/yourdb
```

### 5. Initialize the Database

Run the initialization script to create the necessary tables and indexes:

```bash
export PYTHONPATH=$PYTHONPATH:.
python src/utils/db_init.py
```

## Usage

To run the pipeline manually:

```bash
python main.py
```

The pipeline will:
1. Fetch the latest list of S&P 500 tickers.
2. For each ticker, fetch historical data and calculate technical features.
3. Save new market data to the database.
4. Evaluate any pending predictions from previous days.
5. Generate and save new predictions for the next trading day.

## Deployment on GitHub

1. Push the code to your GitHub repository.
2. Go to **Settings > Secrets and variables > Actions**.
3. Add a new repository secret:
   - `DATABASE_URL`: Your production PostgreSQL connection string.
4. The pipeline will now run automatically every day at midnight UTC. You can also trigger it manually from the **Actions** tab.

## License

MIT