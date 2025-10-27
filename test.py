import pandas as pd

path = "Csv_Files/Raw_Market_Data/market_data_to_check"
file_name = "Raw_Market_Data_06-24-2025.csv"
file_path = f"{path}/{file_name}"
df = pd.read_csv(file_path)

ticker_df = df[df['Ticker'] == 'SOXL']
ticker_df = ticker_df[['Ticker','Price','Atr14','Volatility Percent','Early Morning Atr Warmup Fix','Time']]
ticker_df.to_csv("testing_csv.csv")




