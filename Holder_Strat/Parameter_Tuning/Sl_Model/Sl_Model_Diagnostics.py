import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

version = None
sl_dir = "Holder_Strat/Parameter_Tuning/Sl_Model"
def Set_Version(passed_version):
    global version
    version = passed_version

def Model_Diagnostics(model, results_df, scaler):
    """
    Comprehensive model evaluation for CatBoost trading classifier.
    
    Evaluates:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix
    - Feature importance
    - Prediction distribution
    - Class-specific performance
    - Feature partial dependence (if possible)
    """
    print("="*60)
    print("MODEL DIAGNOSTICS - CatBoost Trading Classifier")
    print("="*60)
    
    # Prepare data
    x_cols = ['minutes since market open', 'volatility percent', 'stop loss']
    results_df = results_df.copy()
    # Handle missing values same as training - drop NaN rows
    results_df['result'] = results_df['result'].replace('NaN', pd.NA)
    results_df = results_df.dropna(subset=['result'])
    
    # Handle float to int conversion (0.0 -> 0, 1.0 -> 1) same as training
    valid_numeric_values = {0, 1, 0.0, 1.0}
    actual_values = set(results_df['result'].unique())
    if not actual_values.issubset(valid_numeric_values):
        results_df = results_df[results_df['result'].isin([0, 1, 0.0, 1.0])]
    results_df['result'] = results_df['result'].astype(int)
    
    X_raw = results_df[x_cols].values
    y = results_df['result'].values
    X_scaled = scaler.transform(X_raw)  # Use transform, not fit_transform
    
    # Check class distribution for stratification
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = min(class_counts)
    
    # Split data (same as training)
    if min_class_count < 2:
        # Use regular split without stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    else:
        # Use stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # 1. BASIC MODEL PERFORMANCE
    print("\n1. BASIC MODEL PERFORMANCE")
    print("-" * 30)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Validation Accuracy: {val_acc:.3f}")
    print(f"Overfitting Check: {train_acc - val_acc:.3f} (< 0.05 is good)")
    
    # 2. DETAILED CLASSIFICATION REPORT
    print("\n2. DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    class_names = ['Stopped Out (0)', 'Reached Target (1)']
    print("Validation Set Performance:")
    print(classification_report(y_val, val_pred, target_names=class_names))
    
    # 3. CONFUSION MATRIX
    print("\n3. CONFUSION MATRIX")
    print("-" * 20)
    cm = confusion_matrix(y_val, val_pred)
    print("Validation Confusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print(f"{'':>12} {'Stop(0)':>8} {'Target(1)':>9}")
    for i, class_name in enumerate(['Stop(0)', 'Target(1)']):
        print(f"{class_name:>12} {cm[i,0]:>8} {cm[i,1]:>9}")
    
    # 4. FEATURE IMPORTANCE
    print("\n4. FEATURE IMPORTANCE")
    print("-" * 25)
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': x_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']:.<35} {row['importance']:>6.1f}")
    
    # 5. CLASS DISTRIBUTION ANALYSIS
    print("\n5. CLASS DISTRIBUTION ANALYSIS")
    print("-" * 35)
    
    # Original distribution
    orig_dist = pd.Series(y).value_counts().sort_index()
    print("Original Data Distribution:")
    for class_val, count in orig_dist.items():
        pct = count / len(y) * 100
        print(f"  Class {class_val} ({class_names[class_val]}): {count:>5} ({pct:>5.1f}%)")
    
    # Prediction distribution
    pred_dist = pd.Series(val_pred).value_counts().sort_index()
    print("\nValidation Predictions Distribution:")
    for class_val, count in pred_dist.items():
        pct = count / len(val_pred) * 100
        print(f"  Class {class_val} ({class_names[class_val]}): {count:>5} ({pct:>5.1f}%)")
    
    # 6. PREDICTION CONFIDENCE ANALYSIS
    print("\n6. PREDICTION CONFIDENCE ANALYSIS")
    print("-" * 40)
    val_proba = model.predict_proba(X_val)
    max_proba = np.max(val_proba, axis=1)
    
    print(f"Average Prediction Confidence: {np.mean(max_proba):.3f}")
    print(f"Min Confidence: {np.min(max_proba):.3f}")
    print(f"Max Confidence: {np.max(max_proba):.3f}")
    
    # Confidence by class
    for class_val in range(2):
        class_mask = val_pred == class_val
        if np.any(class_mask):
            class_conf = max_proba[class_mask]
            print(f"Avg confidence for {class_names[class_val]}: {np.mean(class_conf):.3f}")
    
    # 7. FEATURE STATISTICS
    print("\n7. FEATURE STATISTICS")
    print("-" * 25)
    print("Feature ranges in training data:")
    for i, feature in enumerate(x_cols):
        feature_values = X_raw[:, i]
        print(f"  {feature:.<35} {np.min(feature_values):>8.2f} to {np.max(feature_values):>8.2f}")
    
    # 8. TRADING-SPECIFIC METRICS
    print("\n8. TRADING-SPECIFIC ANALYSIS")
    print("-" * 35)
    
    # Success rate by stop loss value
    print("Success Rate by Stop Loss Value:")
    for sl_val in sorted(results_df['stop loss'].unique()):
        sl_mask = results_df['stop loss'] == sl_val
        sl_results = results_df[sl_mask]['result']
        success_rate = (sl_results == 1).mean()
        total_trades = len(sl_results)
        print(f"  SL {sl_val:>5.1f}%: {success_rate:>6.1%} success ({total_trades:>4} trades)")
    
    # Success rate by time of day
    print("\nSuccess Rate by Time Since Market Open:")
    time_bins = [0, 60, 120, 180, 240, 300, 999]  # 0-1hr, 1-2hr, etc.
    time_labels = ['0-1h', '1-2h', '2-3h', '3-4h', '4-5h', '5h+']
    
    results_df['time_bin'] = pd.cut(results_df['minutes since market open'], 
                                   bins=time_bins, labels=time_labels, include_lowest=True)
    
    for time_label in time_labels:
        time_mask = results_df['time_bin'] == time_label
        if time_mask.sum() > 0:
            time_results = results_df[time_mask]['result']
            success_rate = (time_results == 1).mean()
            total_trades = len(time_results)
            print(f"  {time_label:>4}: {success_rate:>6.1%} success ({total_trades:>4} trades)")
    
    # 9. CREATE VISUALIZATIONS
    print("\n9. GENERATING VISUALIZATIONS...")
    print("-" * 35)
    
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Feature Importance
    plt.subplot(3, 3, 1)
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    
    # Plot 2: Confusion Matrix Heatmap
    plt.subplot(3, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stop', 'Target'],
                yticklabels=['Stop', 'Target'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Plot 3: Prediction Confidence Distribution
    plt.subplot(3, 3, 3)
    plt.hist(max_proba, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Max Probability')
    plt.ylabel('Count')
    
    # Plot 4: Success Rate by Stop Loss
    plt.subplot(3, 3, 4)
    sl_success = []
    sl_values = sorted(results_df['stop loss'].unique())
    for sl_val in sl_values:
        sl_mask = results_df['stop loss'] == sl_val
        success_rate = (results_df[sl_mask]['result'] == 1).mean()
        sl_success.append(success_rate)
    
    plt.plot(sl_values, sl_success, 'bo-')
    plt.title('Success Rate by Stop Loss')
    plt.xlabel('Stop Loss (%)')
    plt.ylabel('Success Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Success Rate by Time of Day
    plt.subplot(3, 3, 5)
    time_success = []
    time_counts = []
    for time_label in time_labels:
        time_mask = results_df['time_bin'] == time_label
        if time_mask.sum() > 0:
            success_rate = (results_df[time_mask]['result'] == 1).mean()
            time_success.append(success_rate)
            time_counts.append(time_mask.sum())
        else:
            time_success.append(0)
            time_counts.append(0)
    
    plt.bar(time_labels, time_success, alpha=0.7)
    plt.title('Success Rate by Time Since Open')
    plt.xlabel('Time Period')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    
    # Plot 6: Volatility vs Success Rate
    plt.subplot(3, 3, 6)
    vol_bins = np.percentile(results_df['volatility percent'], [0, 25, 50, 75, 100])
    results_df['vol_quartile'] = pd.cut(results_df['volatility percent'], bins=vol_bins, 
                                       labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
    
    vol_success = []
    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        vol_mask = results_df['vol_quartile'] == quartile
        if vol_mask.sum() > 0:
            success_rate = (results_df[vol_mask]['result'] == 1).mean()
            vol_success.append(success_rate)
        else:
            vol_success.append(0)
    
    plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], vol_success, alpha=0.7)
    plt.title('Success Rate by Volatility Quartile')
    plt.xlabel('Volatility Quartile')
    plt.ylabel('Success Rate')
    
    # Plot 7: Class Distribution Comparison
    plt.subplot(3, 3, 7)
    x_pos = np.arange(2)
    actual_pcts = [orig_dist[i] / len(y) * 100 for i in range(2)]
    pred_pcts = [pred_dist.get(i, 0) / len(val_pred) * 100 for i in range(2)]
    
    width = 0.35
    plt.bar(x_pos - width/2, actual_pcts, width, label='Actual', alpha=0.7)
    plt.bar(x_pos + width/2, pred_pcts, width, label='Predicted', alpha=0.7)
    plt.title('Class Distribution: Actual vs Predicted')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.xticks(x_pos, ['Stop', 'Target'])
    plt.legend()
    
    # Plot 8: Feature Correlation with Success
    plt.subplot(3, 3, 8)
    success_mask = results_df['result'] == 1
    fail_mask = results_df['result'] == 0
    
    feature_idx = 1  # Volatility percent
    success_vals = results_df[success_mask]['volatility percent']
    fail_vals = results_df[fail_mask]['volatility percent']
    
    plt.hist(success_vals, bins=20, alpha=0.5, label='Success', density=True)
    plt.hist(fail_vals, bins=20, alpha=0.5, label='Stopped Out', density=True)
    plt.title('Volatility Distribution by Outcome')
    plt.xlabel('Volatility Percent')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 9: Model Performance Summary
    plt.subplot(3, 3, 9)
    metrics = ['Train Acc', 'Val Acc', 'Precision', 'Recall', 'F1-Score']
    
    # Calculate weighted averages for multi-class metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_val, val_pred, average='weighted')
    recall = recall_score(y_val, val_pred, average='weighted')
    f1 = f1_score(y_val, val_pred, average='weighted')
    
    values = [train_acc, val_acc, precision, recall, f1]
    colors = ['green' if v > 0.6 else 'orange' if v > 0.5 else 'red' for v in values]
    
    plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Model Performance Summary')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{sl_dir}/Diagnostics/catboost_model_diagnostics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Diagnostic plots saved to: {plot_path}")
    plt.show()
    
    print("\n" + "="*60)
    print("MODEL DIAGNOSTICS COMPLETE")
    print("="*60)


def Give_Model_Test_Input(model, scaler):
    """
    NEW APPROACH: Test the regression model with different entry conditions.
    Input: [minutes_since_open, volatility_percent]
    Output: predicted optimal stop loss
    """
    print("\n" + "="*50)
    print("TESTING OPTIMAL STOP LOSS PREDICTIONS")
    print("="*50)
    
    # Test cases: [minutes_since_open, volatility_percent]
    test_cases = [
        [30, 0.15],   # Early morning, low volatility
        [30, 0.35],   # Early morning, high volatility  
        [120, 0.15],  # Mid-morning, low volatility
        [120, 0.35],  # Mid-morning, high volatility
        [240, 0.15],  # Afternoon, low volatility
        [240, 0.35],  # Afternoon, high volatility
    ]
    
    print("Format: [Minutes Since Open, Volatility %] -> Predicted Optimal Stop Loss")
    print("-" * 70)
    
    for test_input in test_cases:
        input_array = np.array(test_input).reshape(1, -1)  # Reshape to 2D array (1 sample, n features)
        input_scaled = scaler.transform(input_array)  # Scale the input
        predicted_sl = model.predict(input_scaled)[0]  # Get single prediction value
        
        print(f"Input: [{test_input[0]:>3}, {test_input[1]:>5.2f}] -> Predicted SL: {predicted_sl:>6.3f}%")
    
    print("\nInterpretation:")
    print("- More negative values = tighter stop losses")
    print("- Less negative values = looser stop losses") 
    print("- Model learns which conditions need tighter vs looser stops")

