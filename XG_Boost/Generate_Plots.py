import yaml
import xgboost as xgb
import pandas as pd
import Preprocess_Data
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import classification_report
from scipy.cluster.hierarchy import linkage, leaves_list
import os
import sys
import inspect
import seaborn as sns


def Generate_SHAP_Summary_Plot(dest_dir, shap_values, X):
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"{dest_dir}/shap_summary_plot.png")
    print(f"Saved: {dest_dir}/shap_summary_plot.png")


def Generate_Feature_Importance_Plot(dest_dir, model):
    print("Generating feature importance plot...")
    xgb.plot_importance(model, max_num_features=15)
    plt.tight_layout()
    plt.savefig(f"{dest_dir}/feature_importance.png")
    print(f"Saved: {dest_dir}/feature_importance.png")


def Generate_SHAP_Force_Waterfall_Plots(X, shap_values, sample_indices, dest_dir):
    print("Generating SHAP Force Waterfall plots...")

    # For multi-class models, we need to plot each class separately
    n_classes = shap_values.values.shape[2] if len(shap_values.values.shape) == 3 else 1
    class_names = ['Bad (-0.5)', 'Neutral (-0.1/neither)', 'Good (0.9)']
    
    for i, idx in enumerate(sample_indices):
        if idx < len(X):
            if n_classes > 1:
                # Multi-class: plot each class
                for class_idx in range(n_classes):
                    plt.figure(figsize=(12, 6))
                    shap.plots.waterfall(shap_values[idx, :, class_idx], show=False)
                    plt.title(f"SHAP Waterfall Plot - Sample {idx} - Class {class_idx} ({class_names[class_idx]})")
                    plt.tight_layout()
                    plt.savefig(f"{dest_dir}/shap_waterfall_plot_sample_{idx}_class_{class_idx}.png", bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"Saved: {dest_dir}/shap_waterfall_plot_sample_{idx}_class_{class_idx}.png")
            else:
                # Binary classification
                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(shap_values[idx], show=False)
                plt.title(f"SHAP Waterfall Plot - Sample {idx}")
                plt.tight_layout()
                plt.savefig(f"{dest_dir}/shap_waterfall_plot_sample_{idx}.png", bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Saved: {dest_dir}/shap_waterfall_plot_sample_{idx}.png")
    
    return sample_indices
    

# just outputs a useful text file summarizing the waterfall plots
def Generate_SHAP_Force_Waterfall_Summary_Text(shap_values, sample_indices, dest_dir):
    # now make a txt file of the summary of those plots
    shap_matrix = shap_values.values
    feature_names = shap_values.feature_names
    
    # Check if multi-class
    n_classes = shap_matrix.shape[2] if len(shap_matrix.shape) == 3 else 1
    class_names = ['Bad (-0.5)', 'Neutral (-0.1/neither)', 'Good (0.9)']
    
    if n_classes > 1:
        # Multi-class: create summary for each class
        for class_idx in range(n_classes):
            # Extract SHAP values for this class: shape (n_samples, n_features)
            class_shap_matrix = shap_matrix[:, :, class_idx]
            selected_shap_values = class_shap_matrix[sample_indices]

            # find the sum of each feature for their x-axis
            shap_sums = selected_shap_values.sum(axis=0)

            # now find each features average placement in the plots
            rank_matrix = np.zeros((len(sample_indices), len(feature_names)))
            # Loop through each selected sample and rank features
            for row_i, sample_idx in enumerate(sample_indices):
                row_shap = class_shap_matrix[sample_idx]
                
                # Get sorted indices by descending absolute shap value
                sorted_indices = np.argsort(-np.abs(row_shap))
                
                # Assign ranks: rank 1 = highest importance (plot top)
                for rank, feature_idx in enumerate(sorted_indices):
                    rank_matrix[row_i, feature_idx] = rank + 1  # 1-based index

            avg_ranks = rank_matrix.mean(axis=0)

            summary_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Sum': shap_sums,
                'Avg_Rank_Position': avg_ranks
            }).sort_values(by='SHAP_Sum', ascending=False)

            # Save to text file
            output_file = f"{dest_dir}/Summary_Waterfall_Plots_Class_{class_idx}.txt"
            with open(output_file, "w") as f:
                f.write(f"SHAP Feature Summary Across Waterfall Plots - Class {class_idx} ({class_names[class_idx]})\n")
                f.write("=" * 70 + "\n")
                for _, row in summary_df.iterrows():
                    f.write(f"{row['Feature']:<25}: SHAP Sum = {row['SHAP_Sum']:>8.4f}, Avg Rank Position = {row['Avg_Rank_Position']:>5.2f}\n")
    else:
        # Binary classification: original logic
        selected_shap_values = shap_matrix[sample_indices]

        # find the sum of each feature for their x-axis
        shap_sums = selected_shap_values.sum(axis=0)

        # now find each features average placement in the plots
        rank_matrix = np.zeros((len(sample_indices), len(feature_names)))
        # Loop through each selected sample and rank features
        for row_i, sample_idx in enumerate(sample_indices):
            row_shap = shap_matrix[sample_idx]
            
            # Get sorted indices by descending absolute shap value
            sorted_indices = np.argsort(-np.abs(row_shap))
            
            # Assign ranks: rank 1 = highest importance (plot top)
            for rank, feature_idx in enumerate(sorted_indices):
                rank_matrix[row_i, feature_idx] = rank + 1  # 1-based index

        avg_ranks = rank_matrix.mean(axis=0)

        summary_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Sum': shap_sums,
            'Avg_Rank_Position': avg_ranks
        }).sort_values(by='SHAP_Sum', ascending=False)

        # Save to text file
        output_file = f"{dest_dir}/Summary_Waterfall_Plots.txt"
        with open(output_file, "w") as f:
            f.write("SHAP Feature Summary Across Waterfall Plots\n")
            f.write("=" * 50 + "\n")
            for _, row in summary_df.iterrows():
                f.write(f"{row['Feature']:<25}: SHAP Sum = {row['SHAP_Sum']:>8.4f}, Avg Rank Position = {row['Avg_Rank_Position']:>5.2f}\n")


def Generate_SHAP_Interactive_Force_HTML_Plot(shap_values, dest_dir):
    print("Generating interactive force plot...")
    
    # Check if multi-class
    n_classes = shap_values.values.shape[2] if len(shap_values.values.shape) == 3 else 1
    class_names = ['Bad (-0.5)', 'Neutral (-0.1/neither)', 'Good (0.9)']
    
    if n_classes > 1:
        # Multi-class: create force plot for each class
        for class_idx in range(n_classes):
            # For multi-class, we need expected_value[class_idx] and shap_values[..., class_idx]
            force_plot = shap.plots.force(
                base_value=shap_values.base_values[0, class_idx],  # Expected value for this class
                shap_values=shap_values.values[:100, :, class_idx],  # SHAP values for this class
                features=shap_values.data[:100],  # Feature values
                feature_names=shap_values.feature_names
            )
            shap.save_html(f"{dest_dir}/shap_force_plot_class_{class_idx}.html", force_plot)
            print(f"Saved: {dest_dir}/shap_force_plot_class_{class_idx}.html")
    else:
        # Binary classification: use single expected value
        force_plot = shap.plots.force(
            base_value=shap_values.base_values[0],  # Expected value
            shap_values=shap_values.values[:100],  # SHAP values
            features=shap_values.data[:100],  # Feature values
            feature_names=shap_values.feature_names
        )
        shap.save_html(f"{dest_dir}/shap_force_plot.html", force_plot)
        print(f"Saved: {dest_dir}/shap_force_plot.html")
    
    plt.close()


'''
you can do with auto with the top features via
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_features = X.columns[np.argsort(mean_abs_shap)[-10:]].tolist()
IMPORTANT: this has logic to pick the best feature for the coloring (interaction_index = auto). 
           I can override it though via replacing auto with a feature
'''
def Generate_SHAP_dependence_Plot(shap_values, X, dest_dir):
    features_to_plot = ['Entry_Volatility_Percent', 'Entry_Volatility_Ratio', 'Prev_Max_Macd_Diff_Percent', 
                        'Rsi_Extreme_Prev_Cross', 'Entry_My_Adx']
    
    # Check if multi-class
    n_classes = shap_values.values.shape[2] if len(shap_values.values.shape) == 3 else 1
    class_names = ['Bad (-0.5)', 'Neutral (-0.1/neither)', 'Good (0.9)']
    
    for feature in features_to_plot:
        if n_classes > 1:
            # Multi-class: plot for each class
            for class_idx in range(n_classes):
                shap.dependence_plot(
                    ind=feature,              # name of the feature you want to analyze
                    shap_values=shap_values.values[:, :, class_idx],
                    features=X,
                    interaction_index="auto", # can be "auto" or the name of another feature
                    show=False                # Prevents the plot from immediately displaying (which is default behavior).
                )
                plt.title(f"Dependence Plot: {feature} - Class {class_idx} ({class_names[class_idx]})")
                plt.grid(True)       # add grid lines
                plt.tight_layout()   # ensures labels don't get clipped.
                plt.savefig(f"{dest_dir}/shap_dependence_{feature}_class_{class_idx}.png", bbox_inches='tight', dpi=300)
                plt.close()
        else:
            # Binary classification
            shap.dependence_plot(
                ind=feature,              # name of the feature you want to analyze
                shap_values=shap_values.values,
                features=X,
                interaction_index="auto", # can be "auto" or the name of another feature
                show=False                # Prevents the plot from immediately displaying (which is default behavior).
            )
            plt.title(f"Dependence Plot: {feature}")
            plt.grid(True)       # add grid lines
            plt.tight_layout()   # ensures labels don't get clipped.
            plt.savefig(f"{dest_dir}/shap_dependence_{feature}.png", bbox_inches='tight', dpi=300)
            plt.close()


def Generate_SHAP_Value_Correltation(dest_dir, shap_values, X):
    # Check if multi-class
    n_classes = shap_values.values.shape[2] if len(shap_values.values.shape) == 3 else 1
    class_names = ['Bad (-0.5)', 'Neutral (-0.1/neither)', 'Good (0.9)']
    
    if n_classes > 1:
        # Multi-class: create correlation matrix for each class
        for class_idx in range(n_classes):
            shap_df = pd.DataFrame(shap_values.values[:, :, class_idx], columns=X.columns)
            shap_corr = shap_df.corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(shap_corr, cmap='coolwarm', center=0, annot=False)
            plt.title(f"Correlation Matrix of SHAP Values - Class {class_idx} ({class_names[class_idx]})")
            plt.tight_layout()
            plt.savefig(f"{dest_dir}/Shap_Correlation_Plot_class_{class_idx}.png", dpi=300)
            plt.close()
    else:
        # Binary classification
        shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
        shap_corr = shap_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(shap_corr, cmap='coolwarm', center=0, annot=False)
        plt.title("Correlation Matrix of SHAP Values")
        plt.tight_layout()
        plt.savefig(f"{dest_dir}/Shap_Correlation_Plot.png", dpi=300)
        plt.close()


# kernal density estimate
# show correlation between 2 features
def Generate_KDE_plot(dest_dir, X, y):
    featureX = ["Entry_Volatility_Percent", "Entry_Volatility_Percent", "Entry_Volatility_Ratio"]
    featureY = ["Entry_Volatility_Ratio",   "Entry_My_Adx",             "Entry_My_Adx"]

    for i in range(0, len(featureX)):
        plt.figure(figsize=(8, 6))
        sns.kdeplot(
            data=X, 
            x=featureX[i], 
            y=featureY[i],
            fill=True, 
            cmap="magma", 
            thresh=0.05, 
            levels=100
        )
        '''
        This labels it by target, so we see if it succeeds in the dense areas. seems odd to me, like label 1 overwrites the label 0 areas so we're missing data
        sns.kdeplot(
        data=X.assign(label=y),  # Assuming `y` is your actual result column
        x=featureX, y=featureY,
        hue="label",
        fill=True,
        common_norm=False,
        cmap="coolwarm"
        )
        '''

        plt.title(f"Density Plot: {featureX[i]} vs {featureY[i]}")
        plt.xlabel(featureX[i])
        plt.ylabel(featureY[i])
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{dest_dir}/KDE_{featureX[i]}_vs_{featureY[i]}.png", dpi=300)


def Generate_Classification_Report(dest_dir, model, X, y):
    # get predictions
    y_probs = model.predict(X)                      # Probabilities (0.0 to 1.0)
    y_pred = (y_probs > 0.5).astype(int)            # Binary classification

    # --- generate classification report ---
    report = classification_report(y, y_pred, target_names=["Class 0", "Class 1"])

    # --- write to a text file ---
    with open(f"{dest_dir}/ClassificationReport.txt", "w") as f:
        f.write("Classification Report for Full Dataset\n")
        f.write("=" * 50 + "\n")
        f.write(report)

    print(f"Classification report saved to {dest_dir}/ClassificationReport.txt")


