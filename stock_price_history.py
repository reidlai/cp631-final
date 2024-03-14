import yfinance as yf
from datetime import datetime
import csv
import os
import logging
import pandas as pd

class Row:
    def __init__(self, timestamp, open, high, low, close, adjclose, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume

def get_stock_price_history_quotes(stock_symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S%z")
    end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S%z")

    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
    except Exception as e:
        logging.error(f"Symbol not found: {stock_symbol}")
        return []

    quotes = []
    for index, row in data.iterrows():
        quote = Row(index, row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'])
        quotes.append(quote)

    quotes.sort(key=lambda x: x.timestamp)
    
    # convert quotes into dataframe
    quotes_df = pd.DataFrame([vars(quote) for quote in quotes])
    # add symbol column
    quotes_df['symbol'] = stock_symbol
    return quotes_df
    

# def fetch_stock_price_history_quotes(stock_symbol, start_date, end_date, dir_path):
#     # Call get_stock_price_history_quotes
#     stock_price_history_quotes = get_stock_price_history_quotes(stock_symbol, start_date, end_date)

#     # if directory does not exist, create it
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)

#     # Open the CSV file for writing
#     with open(f"{dir_path}/{stock_symbol}.csv", 'w', newline='') as csvfile:
#         fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'adjclose', 'volume']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # Write the header
#         writer.writeheader()

#         # Write the data
#         for quote in stock_price_history_quotes:
#             writer.writerow({
#                 'timestamp': quote.timestamp,
#                 'open': quote.open,
#                 'high': quote.high,
#                 'low': quote.low,
#                 'close': quote.close,
#                 'adjclose': quote.adjclose,
#                 'volume': quote.volume,
#             })