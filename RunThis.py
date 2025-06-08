import Parser
import pandas as pd
import os

# 1) put raw tos csv's in 1tosRawCsvs
# 2) put market data in MarketData
# 3) Running this makes new csv's in 2SummarizedTrades



tosRawCsvDir_todo = '1tosRawCsvs-TODO/'
tosRawCsvDir_done = '1tosRawCsvs-DONE/'
marketDataDir = '2MarketData/'
summarizedTradesDir = '3SummarizedTrades'

headers = ["Date", "Amount", "Entry Time", "Exit Time", "Ticker", "Entry", "Exit", "Shares", "$ Change", "% Change", 
           "Buy/Short", "Technical Type", "High", "Low"]


# using tos raw trade logs, make it readable, then add in market data - save it as summary
# Loop over all CSV files in the directory
#for filename in os.listdir(tosRawCsvDir_todo):
#if filename.endswith('.csv'):
def CreateSummaryCsv(fileName):
        csvPath = os.path.join(tosRawCsvDir_todo, fileName)
        print(f"Processing {fileName}...")

        df = Parser.CreateDf(csvPath)

        tradeDf = Parser.GenerateTradeSummary(df)

        summaryDf = Parser.AddMarketData(tradeDf, tosRawCsvDir_todo, tosRawCsvDir_done)
        
        Parser.AddEstimates(summaryDf, tosRawCsvDir_todo, tosRawCsvDir_done)


testfileName = '2025-04-30-TradeActivity.csv'
CreateSummaryCsv(testfileName)