#!/usr/bin/env python3
"""
Complete Sentiment Analysis Pipeline
====================================

This script takes three parquet files as input and creates a complete sentiment analysis pipeline:
1. Data inspection and loading
2. Text cleaning and preprocessing  
3. Feature extraction using TF-IDF
4. Model training using Logistic Regression
5. Model evaluation and saving

Input files required:
- twitter-sentiment-analysis-train.parquet
- twitter-sentiment-analysis-validation.parquet
- twitter-sentiment-analysis-test.parquet

Output files created:
- train_clean.csv, val_clean.csv, test_clean.csv (cleaned data)
- tfidf_vectorizer.pkl (trained vectorizer)
- label_encoder.pkl (label encoder)
- sentiment_lr_model.pkl (trained model)
- X_train_tfidf.pkl, X_val_tfidf.pkl, X_test_tfidf.pkl (vectorized features)
- y_train.pkl, y_val.pkl, y_test.pkl (encoded labels)
"""

import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import os
import sys

# ===========================================================
#  DIRECTORY SETUP
# ===========================================================
def ensure_directories():
    """
    Create necessary directories if they don't exist
    """
    directories = ['models', 'data', 'temp']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

print("Setting up directories...")
ensure_directories()
print("Directory setup complete.\n")

# ===========================================================
#  STEP 1: DATA INSPECTION
# ===========================================================

def inspect_data(path, dataset_name):
    """
    Inspect a parquet file and identify label column
    """
    print(f"\n{'='*50}")
    print(f"INSPECTING {dataset_name.upper()} DATASET")
    print(f"{'='*50}")
    
    print(f"Loading {path} ...")
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show first 5 rows
    print("\nSample rows:")
    print(df.head().to_string(index=False, max_cols=10))
    
    # Try to guess label column
    label_col = None
    
    # Check for common label column names
    for possible_label in ["sentiment", "label", "target", "class", "feeling"]:
        if possible_label in df.columns:
            labels = df[possible_label].astype(str)
            print(f"\nLabel column detected: '{possible_label}'")
            print("Label distribution:")
            print(labels.value_counts())
            label_col = possible_label
            break
    
    # Fallback: look for small-int columns
    if label_col is None:
        for c in df.columns:
            if df[c].dtype.kind in 'i':
                unique = df[c].nunique()
                if 2 <= unique <= 10:
                    print(f"\nPossible label candidate: '{c}' (integer with {unique} unique values)")
                    print(df[c].value_counts())
                    label_col = c
                    break
    
    if label_col is None:
        print("\nNo obvious label column found. Here are dtypes and sample values:")
        print(df.dtypes)
        for c in df.columns:
            print(f"\nColumn: {c} — sample:")
            print(df[c].dropna().astype(str).head(3).to_string(index=False))
    
    return df, label_col

# ===========================================================
#  STEP 2: TEXT CLEANING
# ===========================================================

def clean_text(text):
    """
    Clean and preprocess text data
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove non-alphanumeric except punctuation
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_and_prepare_data(path, label_col):
    """
    Load parquet file and prepare cleaned data
    """
    df = pd.read_parquet(path)
    
    # Map labels (assuming binary classification with 0/1 or similar)
    if label_col == 'feeling':
        df['label'] = df['feeling'].map({0: "negative", 1: "positive"})
    elif label_col in df.columns:
        # Handle different label formats
        unique_labels = df[label_col].unique()
        if len(unique_labels) == 2:
            if set(unique_labels) == {0, 1}:
                df['label'] = df[label_col].map({0: "negative", 1: "positive"})
            elif set(unique_labels) == {"negative", "positive"}:
                df['label'] = df[label_col]
            else:
                # Map first unique to negative, second to positive
                sorted_labels = sorted(unique_labels)
                df['label'] = df[label_col].map({
                    sorted_labels[0]: "negative", 
                    sorted_labels[1]: "positive"
                })
        else:
            print(f"Warning: More than 2 unique labels found: {unique_labels}")
            # For multi-class, keep original labels
            df['label'] = df[label_col].astype(str)
    
    # Find text column
    text_col = None
    for col in ['text', 'comment', 'review', 'message', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        # Find the column with longest average text
        text_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()
                text_lengths[col] = avg_len
        
        if text_lengths:
            text_col = max(text_lengths, key=text_lengths.get)
            print(f"Assuming '{text_col}' is the text column (longest average length)")
    
    if text_col is None:
        raise ValueError("Could not identify text column")

    # Clean text
    df['clean_text'] = df[text_col].apply(clean_text)

    return df[['clean_text', 'label']]

def fix_cleaned_data(df):
    """
    Fix common issues in cleaned data
    """
    # Replace NaN with empty
    df['clean_text'] = df['clean_text'].fillna("")

    # If clean_text becomes empty, replace with placeholder
    df.loc[df['clean_text'].str.strip() == "", 'clean_text'] = "emptytext"
    
    # Remove rows where label is NaN
    df = df.dropna(subset=['label'])
    
    return df

# ===========================================================
#  STEP 3: FEATURE ENGINEERING
# ===========================================================

def extract_statistical_features(df):
    """
    Extract statistical features from text data
    """
    print("\nExtracting statistical features from text...")
    
    features_df = df.copy()
    
    # Basic text statistics
    features_df['text_length'] = features_df['clean_text'].str.len()
    features_df['word_count'] = features_df['clean_text'].str.split().str.len()
    features_df['char_count'] = features_df['clean_text'].str.len()
    features_df['sentence_count'] = features_df['clean_text'].str.count(r'[.!?]+') + 1
    
    # Average word length
    features_df['avg_word_length'] = features_df['clean_text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
    )
    
    # Punctuation features
    features_df['punctuation_count'] = features_df['clean_text'].str.count(r'[^\w\s]')
    features_df['exclamation_count'] = features_df['clean_text'].str.count('!')
    features_df['question_count'] = features_df['clean_text'].str.count('\?')
    features_df['comma_count'] = features_df['clean_text'].str.count(',')
    features_df['period_count'] = features_df['clean_text'].str.count('\.')
    
    # Uppercase features
    features_df['uppercase_count'] = features_df['clean_text'].apply(
        lambda x: sum(1 for c in x if c.isupper())
    )
    features_df['uppercase_ratio'] = features_df['uppercase_count'] / (features_df['char_count'] + 1)
    
    # Numerical features
    features_df['digit_count'] = features_df['clean_text'].str.count(r'\d')
    features_df['digit_ratio'] = features_df['digit_count'] / (features_df['char_count'] + 1)
    
    # Whitespace features
    features_df['whitespace_count'] = features_df['clean_text'].str.count(r'\s')
    features_df['whitespace_ratio'] = features_df['whitespace_count'] / (features_df['char_count'] + 1)
    
    # Word-level statistics
    word_lengths = features_df['clean_text'].apply(
        lambda x: [len(word) for word in x.split()] if x.split() else [0]
    )
    
    features_df['min_word_length'] = word_lengths.apply(min)
    features_df['max_word_length'] = word_lengths.apply(max)
    features_df['std_word_length'] = word_lengths.apply(
        lambda x: np.std(x) if len(x) > 1 else 0
    )
    
    # Sentence-level statistics (approximate)
    sentence_lengths = features_df['clean_text'].apply(
        lambda x: [len(sent.split()) for sent in re.split(r'[.!?]+', x) if sent.strip()]
    )
    
    features_df['min_sentence_length'] = sentence_lengths.apply(
        lambda x: min(x) if x else 0
    )
    features_df['max_sentence_length'] = sentence_lengths.apply(
        lambda x: max(x) if x else 0
    )
    features_df['mean_sentence_length'] = sentence_lengths.apply(
        lambda x: np.mean(x) if x else 0
    )
    features_df['std_sentence_length'] = sentence_lengths.apply(
        lambda x: np.std(x) if len(x) > 1 else 0
    )
    
    # Vocabulary diversity features
    features_df['unique_word_count'] = features_df['clean_text'].apply(
        lambda x: len(set(x.split())) if x.split() else 0
    )
    features_df['vocabulary_richness'] = features_df['unique_word_count'] / (features_df['word_count'] + 1)
    
    # Readability features
    features_df['avg_words_per_sentence'] = features_df['word_count'] / (features_df['sentence_count'] + 1)
    features_df['complexity_score'] = (
        features_df['avg_word_length'] * features_df['avg_words_per_sentence']
    )
    
    # Select statistical feature columns
    stat_columns = [
        'text_length', 'word_count', 'char_count', 'sentence_count',
        'avg_word_length', 'punctuation_count', 'exclamation_count',
        'question_count', 'comma_count', 'period_count',
        'uppercase_count', 'uppercase_ratio', 'digit_count', 'digit_ratio',
        'whitespace_count', 'whitespace_ratio', 'min_word_length',
        'max_word_length', 'std_word_length', 'min_sentence_length',
        'max_sentence_length', 'mean_sentence_length', 'std_sentence_length',
        'unique_word_count', 'vocabulary_richness', 'avg_words_per_sentence',
        'complexity_score'
    ]
    
    # Handle any inf or nan values
    for col in stat_columns:
        features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
        features_df[col] = features_df[col].fillna(0)
    
    print(f"Extracted {len(stat_columns)} statistical features")
    print("Statistical features:", stat_columns)
    
    return features_df[['clean_text', 'label'] + stat_columns], stat_columns

# ===========================================================
#  STEP 4: VECTORIZATION WITH FEATURE COMBINATION
# ===========================================================

def vectorize_data_with_features(train_df, val_df, test_df, stat_columns):
    """
    Vectorize text data using TF-IDF and combine with statistical features
    """
    print("\n" + "="*50)
    print("VECTORIZING TEXT AND COMBINING FEATURES")
    print("="*50)

    print("Dataset shapes:")
    print(f" Train: {train_df.shape}")
    print(f" Val:   {val_df.shape}")
    print(f" Test:  {test_df.shape}")

    # Remove any remaining NaN rows
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    # TF-IDF Vectorizer for text features
    print("\nFitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=10000,  # Reduced to make room for statistical features
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,
        max_df=0.95
    )
    tfidf.fit(train_df["clean_text"])

    # Transform text to TF-IDF features
    print("Transforming text to TF-IDF features...")
    X_train_tfidf = tfidf.transform(train_df["clean_text"])
    X_val_tfidf = tfidf.transform(val_df["clean_text"])
    X_test_tfidf = tfidf.transform(test_df["clean_text"])

    # Standardize statistical features
    print("Standardizing statistical features...")
    scaler = StandardScaler()
    
    X_train_stats = scaler.fit_transform(train_df[stat_columns])
    X_val_stats = scaler.transform(val_df[stat_columns])
    X_test_stats = scaler.transform(test_df[stat_columns])

    # Combine TF-IDF and statistical features
    print("Combining TF-IDF and statistical features...")
    X_train = hstack([X_train_tfidf, X_train_stats])
    X_val = hstack([X_val_tfidf, X_val_stats])
    X_test = hstack([X_test_tfidf, X_test_stats])

    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_val = le.transform(val_df["label"])
    y_test = le.transform(test_df["label"])

    print(f"\nFinal feature shapes:")
    print(f" X_train: {X_train.shape} (TF-IDF: {X_train_tfidf.shape[1]}, Stats: {X_train_stats.shape[1]})")
    print(f" X_val:   {X_val.shape}")
    print(f" X_test:  {X_test.shape}")
    
    print(f"\nLabel classes: {le.classes_}")
    
    feature_info = {
        'tfidf_features': X_train_tfidf.shape[1],
        'stat_features': X_train_stats.shape[1],
        'total_features': X_train.shape[1],
        'stat_columns': stat_columns
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, tfidf, le, scaler, feature_info

# ===========================================================
#  STEP 5: MODEL TRAINING
# ===========================================================

def train_model(X_train, X_val, y_train, y_val, feature_info):
    """
    Train Logistic Regression model with feature engineering
    """
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*50)

    print(f"Training with {feature_info['total_features']} total features:")
    print(f"  - TF-IDF features: {feature_info['tfidf_features']}")
    print(f"  - Statistical features: {feature_info['stat_features']}")
    
    clf = LogisticRegression(
        max_iter=500,  # Increased for more complex feature space
        n_jobs=-1,
        C=1.0,
        solver="lbfgs",
        random_state=42
    )

    print("Fitting model...")
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Feature importance analysis (for statistical features)
    if hasattr(clf, 'coef_'):
        print("\nTop 10 Statistical Feature Importances:")
        # Get coefficients for statistical features (last part of feature vector)
        stat_coefs = clf.coef_[0][-feature_info['stat_features']:]
        stat_importance = list(zip(feature_info['stat_columns'], np.abs(stat_coefs)))
        stat_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(stat_importance[:10]):
            print(f"  {i+1:2d}. {feature:<25}: {importance:.4f}")

    return clf

# ===========================================================
#  STEP 6: SAVE ALL OUTPUTS
# ===========================================================

def save_all_outputs(train_df, val_df, test_df, X_train, X_val, X_test, 
                    y_train, y_val, y_test, tfidf, le, scaler, clf, feature_info):
    """
    Save all processed data and trained models
    """
    print("\n" + "="*50)
    print("SAVING ALL OUTPUTS")
    print("="*50)

    # Save all preprocessors
    pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl", "wb"))
    pickle.dump(le, open("models/label_encoder.pkl", "wb"))
    pickle.dump(scaler, open("models/feature_scaler.pkl", "wb"))
    pickle.dump(feature_info, open("models/feature_info.pkl", "wb"))
    print("✓ Saved preprocessors: tfidf_vectorizer.pkl, label_encoder.pkl, feature_scaler.pkl, feature_info.pkl")

    # Save trained model
    pickle.dump(clf, open("models/sentiment_lr_model_enhanced.pkl", "wb"))
    print("✓ Saved enhanced model: sentiment_lr_model_enhanced.pkl")
    
    # Also save a simple model for backwards compatibility
    pickle.dump(clf, open("models/sentiment_lr_model.pkl", "wb"))
    print("✓ Saved model (compatibility): sentiment_lr_model.pkl")

# ===========================================================
#  MAIN PIPELINE
# ===========================================================

def main():
    """
    Main pipeline function
    """
    print("="*60)
    print("COMPLETE SENTIMENT ANALYSIS PIPELINE")
    print("="*60)

    # File paths
    files = {
        "train": "data/twitter-sentiment-analysis-train.parquet",
        "val": "data/twitter-sentiment-analysis-validation.parquet", 
        "test": "data/twitter-sentiment-analysis-test.parquet"
    }
    
    # Check if input files exist - try both data folder and current directory
    missing_files = []
    for name, path in files.items():
        if not os.path.exists(path):
            # Try current directory as fallback
            fallback_path = path.replace("data/", "")
            if os.path.exists(fallback_path):
                files[name] = fallback_path
                print(f"Found {fallback_path} in current directory (moved to data/ is recommended)")
            else:
                missing_files.append(path)
    
    if missing_files:
        print("ERROR: Input files not found:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease place the parquet files in either:")
        print("  1. data/ folder (recommended)")
        print("  2. Current directory")
        print("\nRequired files:")
        print("  - twitter-sentiment-analysis-train.parquet")
        print("  - twitter-sentiment-analysis-validation.parquet")
        print("  - twitter-sentiment-analysis-test.parquet")
        return
    
    try:
        # Step 1: Inspect and load data
        datasets = {}
        label_cols = {}
        
        for name, path in files.items():
            df, label_col = inspect_data(path, name)
            if df is None:
                return
            datasets[name] = df
            label_cols[name] = label_col
        
        # Verify we found label columns
        if not all(label_cols.values()):
            print("\nERROR: Could not identify label columns in all datasets")
            return
        
        # Use the label column from training data for all datasets
        main_label_col = label_cols['train']
        print(f"\nUsing '{main_label_col}' as label column for all datasets")
        
        # Step 2: Clean and prepare data
        print("\n" + "="*50)
        print("CLEANING AND PREPARING DATA")
        print("="*50)
        
        cleaned_datasets = {}
        for name, df in datasets.items():
            print(f"\nProcessing {name} dataset...")
            cleaned_df = load_and_prepare_data(files[name], main_label_col)
            cleaned_df = fix_cleaned_data(cleaned_df)
            cleaned_datasets[name] = cleaned_df
            print(f"Cleaned {name} shape: {cleaned_df.shape}")
            print(f"Label distribution:\n{cleaned_df['label'].value_counts()}")
        
        train_df = cleaned_datasets['train']
        val_df = cleaned_datasets['val'] 
        test_df = cleaned_datasets['test']
        
        # Step 3: Feature Engineering
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        train_df_features, stat_columns = extract_statistical_features(train_df)
        val_df_features, _ = extract_statistical_features(val_df)
        test_df_features, _ = extract_statistical_features(test_df)
        
        print(f"\nFeature engineering completed:")
        print(f"  - Original features: text + label")
        print(f"  - Added statistical features: {len(stat_columns)}")
        print(f"  - Total columns: {train_df_features.shape[1]}")
        
        # Step 4: Vectorize and combine features
        X_train, X_val, X_test, y_train, y_val, y_test, tfidf, le, scaler, feature_info = vectorize_data_with_features(
            train_df_features, val_df_features, test_df_features, stat_columns
        )
        
        # Step 5: Train model
        clf = train_model(X_train, X_val, y_train, y_val, feature_info)
        
        # Step 6: Save everything
        save_all_outputs(train_df_features, val_df_features, test_df_features, X_train, X_val, X_test,
                        y_train, y_val, y_test, tfidf, le, scaler, clf, feature_info)
        
        print("="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nFinal Model Performance:")
        print(f"  - Total Features: {feature_info['total_features']}")
        print(f"    * TF-IDF Features: {feature_info['tfidf_features']}")
        print(f"    * Statistical Features: {feature_info['stat_features']}")
        print(f"\nGenerated Files:")
        print("  📊 Enhanced Model:")
        print("    - sentiment_lr_model_enhanced.pkl (model with statistical features)")
        print("    - feature_scaler.pkl (statistical feature scaler)")
        print("    - feature_info.pkl (feature metadata)")
        print("  🔧 Core Components:")
        print("    - sentiment_lr_model.pkl (backward compatible)")
        print("    - tfidf_vectorizer.pkl (text vectorizer)")
        print("    - label_encoder.pkl (label encoder)")
        print("  📈 Data Files:")
        print("    - *_clean_features.csv (data with statistical features)")
        print("    - *_clean.csv (basic cleaned data)")
        print("\n🚀 Ready for deployment with enhanced feature engineering!")
        
    except Exception as e:
        print(f"\nERROR during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()