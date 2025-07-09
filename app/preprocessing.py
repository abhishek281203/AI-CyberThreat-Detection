import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
import os
import joblib

# Define constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

def clean_column_names(df):
    """
    Remove spaces from column names and convert to lowercase for consistency
    """
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df

def identify_numeric_features(df):
    """
    Identify numeric features that can be processed
    """
    # Try to convert all columns to numeric, errors='coerce' will convert non-numeric to NaN
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    
    # Identify columns where conversion succeeded (did not create all NaNs)
    valid_numeric = numeric_df.columns[~numeric_df.isna().all()].tolist()
    
    return valid_numeric

def handle_missing_values(df):
    """
    Handle missing and infinite values in the dataset
    """
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Identify numeric columns
    numeric_columns = identify_numeric_features(df)
    print(f"Identified {len(numeric_columns)} numeric columns out of {len(df.columns)} total columns")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Filter only numeric columns for imputation
    df_numeric = df_processed[numeric_columns]
    
    # Check if imputer already exists, otherwise create a new one
    if os.path.exists(IMPUTER_PATH):
        imputer = joblib.load(IMPUTER_PATH)
        imputed_values = imputer.transform(df_numeric)
        df_numeric_imputed = pd.DataFrame(imputed_values, columns=df_numeric.columns)
    else:
        # Create and fit imputer
        imputer = SimpleImputer(strategy='mean')
        imputed_values = imputer.fit_transform(df_numeric)
        df_numeric_imputed = pd.DataFrame(imputed_values, columns=df_numeric.columns)
        
        # Save imputer for future use
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(imputer, IMPUTER_PATH)
    
    # Update the numeric columns in the processed dataframe
    for col in df_numeric.columns:
        df_processed[col] = df_numeric_imputed[col]
    
    return df_processed[numeric_columns]  # Return only the numeric columns

def normalize_features(df, is_training=False):
    """
    Normalize feature values using MinMaxScaler
    """
    # Check if scaler already exists
    if os.path.exists(SCALER_PATH) and not is_training:
        # Load existing scaler for inference
        scaler = joblib.load(SCALER_PATH)
        df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns)
    else:
        # Create and fit a new scaler for training
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        # Save the scaler for future use
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    
    return df_normalized

def encode_labels(df):
    """
    Encode Label column as binary classes (0 = Normal, 1 = Attack)
    """
    if 'label' in df.columns:
        # Map any attack type to 1, and 'BENIGN' or 'Normal' to 0
        df['label'] = df['label'].apply(lambda x: 0 if str(x).upper() in ['BENIGN', 'NORMAL'] else 1)
    
    return df

def preprocess_dataset(df, is_training=True):
    """
    Full preprocessing pipeline for the dataset
    """
    print("Cleaning column names...")
    # Clean column names
    df = clean_column_names(df)
    
    print("Processing labels...")
    # Extract label column if it exists
    if 'label' in df.columns and is_training:
        # Create a separate series for the labels
        labels = df['label'].copy()
        features = df.drop(columns=['label'])
        
        # Encode labels
        labels = pd.Series(labels).apply(lambda x: 0 if str(x).upper() in ['BENIGN', 'NORMAL'] else 1)
    else:
        labels = None
        features = df.copy()
    
    print("Handling missing values and non-numeric data...")
    # Handle missing values and keep only numeric features
    features = handle_missing_values(features)
    
    # Save feature names for inference
    if is_training:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(FEATURE_NAMES_PATH, 'wb') as f:
            pickle.dump(features.columns.tolist(), f)
        print(f"Saved {len(features.columns)} feature names for future reference")
    
    print("Normalizing features...")
    # Normalize features
    features = normalize_features(features, is_training)
    
    # Return preprocessed features and labels
    if is_training:
        return features, labels
    else:
        return features

def preprocess_input_data(input_df):
    """
    Preprocess input data for prediction
    """
    # Clean column names
    input_df = clean_column_names(input_df)
    
    # Load feature names used during training
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        
        # Ensure input data has all required features
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features
        
        # Keep only the features that were used for training
        input_df = input_df[feature_names]
    
    # Handle missing values
    input_df = handle_missing_values(input_df)
    
    # Normalize features
    input_df = normalize_features(input_df, is_training=False)
    
    return input_df 