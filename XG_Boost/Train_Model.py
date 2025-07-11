import yaml
import xgboost as xgb
import pandas as pd
import Preprocess_Data
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, leaves_list
import os
import sys
import inspect
import Generate_Plots as Generate_Plots

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))
data_file_name = 'XG_Boost/Data.csv'

output_dir = 'XG_Boost/Output'
overview_dir = f"{output_dir}/1_overview"
interactions_ranges_dir = f"{output_dir}/2_interactions_and_ranges"
correlations_dir = f"{output_dir}/3_correlations"
datapoint_analysis_dir = f"{output_dir}/4_datapoint_analysis"

def LoadConfig(path="XG_Boost/Config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def Train_Model():
    # --- Load config ---
    config = LoadConfig()

    # --- Preprocess data ---
    X, y = Preprocess_Data.Clean_Data(data_file_name)

    # --- Train the model ---
    model = xgb.XGBClassifier(**config)   # **config means dictionary unpacking, each key in config dictionary becomes an argument
    model.fit(X, y)

    # --- Save model (optional) ---
    model.save_model(f"{output_dir}/trained_model.json")

    return model, X, y


def Load_Pretrained_Model_JSON():
    model_path = r"XG_Boost\Output\trained_model.json"
    X, y = Preprocess_Data.Clean_Data(data_file_name)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    return model, X, y


def Main():
    #model, X, y = Train_Model()
    model, X, y = Load_Pretrained_Model_JSON()

    # --- get shap_values ---
    print("Calculating SHAP values...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # --- SHAP summary plot ---
    Generate_Plots.Generate_SHAP_Summary_Plot(overview_dir, shap_values, X)

    # --- Feature importance plot ---
    Generate_Plots.Generate_Feature_Importance_Plot(overview_dir, model)

    # --- SHAP waterfall plots (multiple) ---
    sample_indices = list(range(29, 34)) # this excludes the final index #29-43
    Generate_Plots.Generate_SHAP_Force_Waterfall_Plots(X, shap_values, sample_indices, datapoint_analysis_dir)
    Generate_Plots.Generate_SHAP_Force_Waterfall_Summary_Text(shap_values, sample_indices, datapoint_analysis_dir)

    # --- Interactive force plot (HTML) ---
    Generate_Plots.Generate_SHAP_Interactive_Force_HTML_Plot(shap_values, datapoint_analysis_dir)

    # --- SHAP dependence plots (multiple) ---
    Generate_Plots.Generate_SHAP_dependence_Plot(shap_values, X, interactions_ranges_dir)

    # --- SHAP Value Correlation Matrix ---
    Generate_Plots.Generate_SHAP_Value_Correltation(correlations_dir, shap_values, X)

    # --- kernal density estimate plot (correlation btw 2 features (kinda like a heatmap)) ---
    Generate_Plots.Generate_KDE_plot(interactions_ranges_dir, X, y)

    # --- Classification report txt ---
    # bad, 100% accuracy since my data isn't prediction ready
    #Generate_Plots.Generate_Classification_Report(output_dir, model, X, y)

    print("Training complete.")







if __name__ == "__main__":
    Main()
