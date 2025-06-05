# =======================================================
# 6. AI-Driven Trend Forecasting
# Goal: Predict the next viral trends on TikTok using ML
# =======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
from IPython.display import display # Added for displaying DataFrames in Jupyter-like environments

# --- 1. Load Data ---
# Assuming the CSV file is in the correct path relative to the script
try:
    df = pd.read_csv('data/processed/tiktok_processed_with_nlp_features.csv', parse_dates=['create_time'])
except FileNotFoundError:
    print("Error: The data file 'data/processed/tiktok_processed_with_nlp_features.csv' was not found.")
    print("Please ensure the file path is correct.")
    exit()

df['clean_description'] = df['clean_description'].fillna("").astype(str)
df['create_time'] = pd.to_datetime(df['create_time'])
df = df.sort_values('create_time')

# --- 2. Define Features ---
numerical_features = [
    'likes', 'comments', 'shares', 'plays', 'hashtag_count',
    'description_length', 'sentiment_polarity'
]
categorical_features = ['author', 'day_of_week', 'is_weekend', 'time_period'] # Ensure 'author' does not have too many unique values for OHE
text_feature = 'clean_description'

# --- 3. Define Viral Threshold ---
# Using a quantile-based threshold for defining 'viral'
threshold = df['virality_score_normalized'].quantile(0.80)
df['is_viral'] = (df['virality_score_normalized'] >= threshold).astype(int)

# --- 4. Label Future Viral Videos (24h and 7d) ---
def label_viral_next_period(df, period_hours):
    df_sorted = df.sort_values('create_time').copy() # Use .copy() to avoid SettingWithCopyWarning
    # Calculate future virality and time for videos by the same author
    df_sorted['future_virality'] = df_sorted.groupby('author')['virality_score_normalized'].shift(-1)
    df_sorted['future_time'] = df_sorted.groupby('author')['create_time'].shift(-1)
    
    # Calculate time difference in hours
    df_sorted['delta_hours'] = (df_sorted['future_time'] - df_sorted['create_time']).dt.total_seconds() / 3600
    
    # Mask for videos within the specified future period
    mask = (df_sorted['delta_hours'] > 0) & (df_sorted['delta_hours'] <= period_hours)
    
    # Label if the future video (within the period) is viral
    df_sorted['is_viral_next'] = ((df_sorted['future_virality'] >= threshold) & mask).astype(int)
    return df_sorted

df_24h = label_viral_next_period(df, 24)
df_7d = label_viral_next_period(df, 24*7)

# --- 5. Preprocessing Pipeline ---
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('txt', TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1,2)), text_feature), # Increased max_features, added ngrams, added text_feature
], remainder='drop') # Explicitly drop other columns

# Feature Selector
selector = SelectKBest(score_func=f_classif, k=50) # Increased k, ensure k <= total features after preprocessing

# --- 6. Trend Forecasting Function (with GridSearchCV and Feature Importance) ---
def run_trend_forecasting(df_labeled, target_col, preprocessor, selector):
    feature_cols = numerical_features + categorical_features + [text_feature]
    
    # Drop rows where the target is NaN (e.g., last videos of an author for future labeling)
    df_valid = df_labeled[df_labeled[target_col].notna()].copy()
    if df_valid.empty:
        print(f"No valid data for target {target_col} after filtering NaNs. Skipping.")
        return pd.DataFrame()

    X = df_valid[feature_cols]
    y = df_valid[target_col]

    if len(y.unique()) < 2:
        print(f"Target column {target_col} has less than 2 unique values. Skipping model training.")
        return pd.DataFrame()
    
    # Stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing (ColumnTransformer)
    try:
        # For scikit-learn >= 1.0, get_feature_names_out is preferred
        feature_names_transformed = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older scikit-learn versions (manual construction can be complex)
        # This is a simplified fallback; robust construction needs care
        num_names = numerical_features
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        txt_names = preprocessor.named_transformers_['txt'].get_feature_names_out()
        feature_names_transformed = list(num_names) + list(cat_names) + list(txt_names)


    # Handle class imbalance using SMOTETomek
    # Apply SMOTE only if there are enough samples in the minority class for k_neighbors
    minority_class_count = y_train.value_counts().min()
    if minority_class_count > 1 and minority_class_count >= 5 : # SMOTE default k_neighbors is 5
        smote = SMOTETomek(random_state=42, n_jobs=-1)
        X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    else:
        print(f"Skipping SMOTETomek for {target_col} due to insufficient minority samples ({minority_class_count}). Using original training data.")
        X_train_res, y_train_res = X_train_proc, y_train


    # Feature selection (ensure k is not greater than number of features)
    current_k = min(selector.k, X_train_res.shape[1])
    if selector.k > X_train_res.shape[1]:
        print(f"Warning: k for SelectKBest ({selector.k}) is > num features ({X_train_res.shape[1]}). Using k={current_k}.")
    
    current_selector = SelectKBest(score_func=f_classif, k=current_k)
    current_selector.fit(X_train_res, y_train_res)
    X_train_sel = current_selector.transform(X_train_res)
    X_test_sel = current_selector.transform(X_test_proc) # Transform test set using already fitted selector

    # Get selected feature names
    selected_indices = current_selector.get_support(indices=True)
    selected_feature_names = [feature_names_transformed[i] for i in selected_indices]

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 150],       # Number of trees
        'max_depth': [7, 10, None],       # Max depth of trees
        'min_samples_split': [2, 5],    # Min samples to split a node
        'min_samples_leaf': [1, 3],     # Min samples at a leaf node
        'class_weight': ['balanced', 'balanced_subsample', None] # Address imbalance if SMOTE isn't perfect
    }
    
    # Use a simpler grid if computation time is an issue
    # param_grid = {
    #     'n_estimators': [100], 'max_depth': [7], 'class_weight': ['balanced']
    # }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3, # 3-fold cross-validation, can be increased to 5
        scoring='roc_auc', # Focus on ROC AUC for imbalanced classes
        n_jobs=-1, # Use all available cores for GridSearchCV
        verbose=1 # Shows progress
    )
    
    grid_search.fit(X_train_sel, y_train_res)
    
    print(f"\nBest parameters found for {target_col}: {grid_search.best_params_}")
    model = grid_search.best_estimator_

    # Predictions and Probabilities
    y_proba = model.predict_proba(X_test_sel)[:, 1]
    
    # Adjust prediction threshold (can be optimized based on Precision-Recall curve)
    prediction_threshold = 0.4 # Adjusted from 0.3, this is a hyperparameter to tune
    y_pred = (y_proba >= prediction_threshold).astype(int)

    # Evaluation
    print(f"\n--- Evaluation for {target_col} (Prediction Threshold: {prediction_threshold}) ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature Importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        # Use display for better formatting in Jupyter, otherwise print
        try:
            display(feature_importance_df.head(10))
        except NameError:
            print(feature_importance_df.head(10))

    # ROC Curve Plot
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.3f}')
    plt.plot([0, 1], [0, 1], 'k--') # Diagonal reference line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {target_col}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output: Top viral predictions as DataFrame
    X_test_original_indices = y_test.index # Get original indices from y_test
    viral_videos_info = df_valid.loc[X_test_original_indices].copy()
    viral_videos_info['predicted_proba'] = y_proba
    viral_videos_info['predicted_label'] = y_pred
    
    # Select relevant columns for the output
    output_columns = ['video_id', 'create_time', 'clean_description', 'author', 'hashtag_list', 'hashtag_count', 'predicted_proba', 'is_viral']
    # Ensure all output columns exist in viral_videos_info
    output_columns = [col for col in output_columns if col in viral_videos_info.columns]

    viral_output = viral_videos_info[viral_videos_info['predicted_label'] == 1][output_columns]
    viral_output = viral_output.sort_values(by='predicted_proba', ascending=False).head(20) # Show top 20
    
    return viral_output

# --- 7. Filter Valid Labeled Rows ---
# Ensure target column 'is_viral_next' is not NaN before passing to the function
df_24h_valid = df_24h[df_24h['is_viral_next'].notna()].copy()
df_7d_valid = df_7d[df_7d['is_viral_next'].notna()].copy()


# --- 8. Run Forecasting for 24h and 7d ---
print("Starting Trend Forecasting for Next 24 Hours...")
top_viral_24h = run_trend_forecasting(df_24h_valid, 'is_viral_next', preprocessor, selector)

print("\n\nStarting Trend Forecasting for Next 7 Days...")
top_viral_7d = run_trend_forecasting(df_7d_valid, 'is_viral_next', preprocessor, selector)

# --- 9. Display Results ---
print("\n--- Top Viral Predictions (Next 24 Hours) ---")
if not top_viral_24h.empty:
    try:
        display(top_viral_24h)
    except NameError:
        print(top_viral_24h)
else:
    print("No viral predictions for the next 24 hours based on the model.")

print("\n--- Top Viral Predictions (Next 7 Days) ---")
if not top_viral_7d.empty:
    try:
        display(top_viral_7d)
    except NameError:
        print(top_viral_7d)
else:
    print("No viral predictions for the next 7 days based on the model.")

print("\nScript execution finished.")