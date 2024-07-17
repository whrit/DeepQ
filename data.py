import yfinance as yf
from datetime import datetime, timedelta
import pytz

# Define the ticker symbol and the time periods
ticker_symbol = 'SPY'  # Replace with your desired ticker symbol

# Define timezone
# timezone = 'America/New_York'

# Get the current time and adjust to the required timezone
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)

# Fetch the data
data = yf.download(ticker_symbol, start=start_date, end=end_date)
# Ensure last_year_date is timezone-aware
last_year_date = end_date - timedelta(days=365*2)


# Split the data into the last year and the previous 9 years
data_last_year = data[data.index >= last_year_date]
data_previous_9_years = data[data.index < last_year_date]

# Save the data to CSV files
data_last_year.to_csv(f'{ticker_symbol}_last_year.csv')
data_previous_9_years.to_csv(f'{ticker_symbol}_previous_9_years.csv')

print("Data saved to CSV files successfully.")
