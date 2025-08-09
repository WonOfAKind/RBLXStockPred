import os
import time

import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone
from pygooglenews import GoogleNews

from sec_edgar_downloader import Downloader

# Sentiment analysis
import finnhub
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Directory constants ---
DATA_DIR = "../data/"
ROMONITOR_DIR = os.path.join(DATA_DIR, "romonitor_data")
SEC_DIR = os.path.join(DATA_DIR, "sec-edgar-filings")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ROMONITOR_DIR, exist_ok=True)
os.makedirs(SEC_DIR, exist_ok=True)

# 1. Market Context Data
def update_market_context():
    tickers = ['^GSPC', '^IXIC', '^VIX', '^TNX', 'XLC']
    # Use 1 year for context
    end = datetime.today()
    start = "2021-03-10"
    context = yf.download(tickers, start=start, end=end)['Close'].reset_index()
    context['date'] = context['Date'].dt.strftime('%Y-%m-%d')
    context = context.drop(columns=['Date'])
    context.to_csv(os.path.join(DATA_DIR, 'market_context_data.csv'), index=False)
    print("Updated market context data.")

# 2. Technical/Price Data
def update_technical():
    tickers = ["RBLX", "^GSPC", "^IXIC"]
    start_date = "2021-03-10"
    data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True)
    rblx = data['RBLX'].copy()
    # Add pandas_ta indicators if you want
    rblx.to_csv(os.path.join(DATA_DIR, 'RBLX_with_technicals.csv'))
    data['^GSPC'].to_csv(os.path.join(DATA_DIR, 'SP500.csv'))
    data['^IXIC'].to_csv(os.path.join(DATA_DIR, 'Nasdaq.csv'))

    print("Updated technical/price data.")

# 3. Sentiment/News Data (Finnhub)
def update_sentiment():
    api_key = os.environ.get('FINNHUB_KEY')
    client = finnhub.Client(api_key=api_key)

    DATA_FILE = os.path.join(DATA_DIR, 'clean_news_sentiment.csv')
    TARGET_FROM_DATE = datetime(2025, 6, 10)
    INITIAL_TO_DATE = datetime.today()  # Start from most recent

    # Helper to get the last date in the CSV (as datetime), or return INITIAL_TO_DATE if file doesn't exist
    def get_last_date():
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            if not df.empty:
                min_date_str = df['date'].min()
                return datetime.strptime(min_date_str, '%Y-%m-%d')
        return INITIAL_TO_DATE

    while True:
        last_date = get_last_date()
        if last_date <= TARGET_FROM_DATE:
            print("All news up to target date fetched.")
            break

        from_date = max(TARGET_FROM_DATE, last_date - timedelta(days=30))  # Fetch up to 30 days at a time
        to_date = last_date

        print(f"Fetching news from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")

        news = client.company_news(
            "RBLX",
            _from=from_date.strftime("%Y-%m-%d"),
            to=to_date.strftime("%Y-%m-%d")
        )

        if not news:
            print("No more news to fetch.")
            break

        sia = SentimentIntensityAnalyzer()
        results = []
        for article in news:
            text = (article.get("headline", "") + "\n" + article.get("summary", "")).strip()
            vs = sia.polarity_scores(text)
            results.append({
                "date": datetime.fromtimestamp(article["datetime"]).strftime("%Y-%m-%d"),
                "headline": article["headline"],
                **vs
            })
        new_df = pd.DataFrame(results)

        # If file exists, append, else just save new
        if os.path.exists(DATA_FILE):
            old_df = pd.read_csv(DATA_FILE)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["date", "headline"], inplace=True)
        else:
            combined_df = new_df

        combined_df.to_csv(DATA_FILE, index=False)
        print(f"Appended news from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}.")

        # To avoid too many requests per minute; adjust as needed to respect API rate limits
        import time
        time.sleep(60)

    print("Finished fetching all available news.")


# 4. User/Engagement Data (RoMonitor API)
def update_romonitor():
    # Always use UTC for timestamps in this API
    today = datetime.now(timezone.utc).strftime('%Y-%m-%dT23:59:59.999Z')

    endpoints = {
        "ccu": f"https://romonitorstats.com/api/v1/charts/get?name=platform-ccus&timeslice=day&start=2020-01-05T00:00:00.000Z&ends={today}",
        "registrations": f"https://romonitorstats.com/api/v1/charts/get?name=platform-registrations&timeslice=day&start=2020-03-15T00:00:00.000Z&ends={today}",
        "session_length": f"https://romonitorstats.com/api/v1/charts/get?name=platform-session-length&timeslice=day&start=2023-04-01T00:00:00.000Z&ends={today}"
    }
    output_dir = os.path.join("../data/", "romonitor_data")
    os.makedirs(output_dir, exist_ok=True)
    for key, url in endpoints.items():
        df = pd.DataFrame(requests.get(url).json())
        df.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)
        print(f"Updated {key} data from RoMonitor.")

# 5. Update and retrieve SEC data
def update_sec_data():
    dl = Downloader("RBLX_SEC", "saysaygo@gmail.com", "../data")
    CIK = "RBLX"

    dl.get("10-K", CIK, after="2021-01-01")
    dl.get("10-Q", CIK, after="2021-01-01")
    dl.get("8-K", CIK, after="2021-01-01")

    print("Updated SEC data")

# 6. Update Quarterly Financials
def update_quarterly_financials():
    rblx = yf.Ticker("RBLX")

    rblx.quarterly_financials.to_csv(DATA_DIR + "RBLX_quarterly_income_statement.csv")
    rblx.quarterly_balance_sheet.to_csv(DATA_DIR + "RBLX_quarterly_balance_sheet.csv")
    rblx.quarterly_cashflow.to_csv(DATA_DIR + "RBLX_quarterly_cashflow.csv")

    print("Updated Quarterly Financials")

if __name__ == "__main__":
    update_market_context()
    update_technical()
    update_sentiment()
    update_romonitor()
    #update_sec_data()
    print("All data updated!")
