import pandas as pd
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt


'''
PURPOSE OF FILE:
Tests metrics for if their logic/info overlaps with other metrics and thus isn't useful
'''




# basically, pairwaise correlation/mutual info btw absolute bias, vol percent and vol ratio
# Pearson & Spearman Correlation: Look for correlations > 0.7 or < -0.7 as a red flag for redundancy
# Mutual Information: 0 = no relationship, >0.1 = some, > 0.3-0.5 = risky (high chance of redundancy)
def Check_logical_overlap_of_volatility_metrics():
    data_dir = "Csv_Files/3_Final_Trade_Csvs"
    data_file = "Bulk_Combined.csv"
    df = pd.read_csv(f"{data_dir}/{data_file}")

    # 1) clean data
    # drop all columns besides 'Entry Directional Bias Abs Distance', 'Entry Volatility Percent', 'Entry Volatility Ratio
    # drop all rows containing an NaN value
    print(f"Data shape before cleaning: {df.shape}")
    columns_to_keep = ['Entry Directional Bias Abs Distance', 'Entry Volatility Percent', 'Entry Volatility Ratio']
    df = df[columns_to_keep]
    df = df.dropna()
    
    print(f"Data shape after cleaning: {df.shape}")
    print(f"Columns kept: {list(df.columns)}")

    # 2) Pearson & Spearman Correlation
    # pearson (linear relationship)
    correlation_matrix = df.corr(method="pearson")
    print("Pearson correlation:\n", correlation_matrix)
    
    # spearman (rank-based — better for nonlinear financial data):
    spearman_matrix = df.corr(method="spearman")
    print("\nSpearman correlation:\n", spearman_matrix)

    # 3) Mutual Information (nonlinear dependency)
    X = df[['Entry Volatility Percent', 'Entry Volatility Ratio']]
    y = df['Entry Directional Bias Abs Distance']

    mi_scores = mutual_info_regression(X, y, discrete_features=False)
    print("\nMutual Information with AbsoluteBias:")
    for feature, score in zip(X.columns, mi_scores):
        print(f"{feature}: {score:.4f}")

    # 4) not useful, but a pairwise plot and scatter plot
    #sns.pairplot(df)
    #plt.show()

    sns.scatterplot(x='Entry Volatility Percent', y="Entry Directional Bias Abs Distance", data=df)
    plt.show()



# tells us if a metric is duplicate info on other metrics
Check_logical_overlap_of_volatility_metrics()

