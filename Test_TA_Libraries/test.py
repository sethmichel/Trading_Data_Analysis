import os
import pandas as pd
import pandas_ta as ta


def Test_Pandas_Ta():
    df = pd.DataFrame()
    df = pd.read_csv("testData.csv")
    tickers = ['HOOD', 'IONQ', 'MARA', 'RDDT', 'SMCI', 'SOXL', 'TSLA']
    
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker].copy()

help(ta.rsi)