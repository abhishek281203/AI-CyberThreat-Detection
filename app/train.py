import pandas as pd
import numpy as np
import os
import pickle
import time
import traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import sys

# Add the app directory to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.preprocessing import preprocess_dataset

# Define paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CICIDS2017.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def inspect_data(df, max_rows=5):
    """
    Inspect the dataset and print useful information
    """
    print("\nData Inspection:")
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    
    # Check data types
    print("\nData types:")
    print(df.dtypes.value_counts())
    
    # Check for missing values
    print("\nMissing values per column (top 10):")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).head(10))
    
    # Print a few sample rows
    print(f"\nSample data (first {max_rows} rows):")
    print(df.head(max_rows).to_string())
    
    # Check for label distribution if 'Label' column exists
    if 'label' in df.columns:
        print("\nLabel distribution:")
        print(df['label'].value_counts())

def load_and_preprocess_data(data_path, sample_size=None):
    """
    Load and preprocess the dataset
    
    Parameters:
    - data_path: Path to the dataset
    - sample_size: Number of rows to sample for faster processing (None for all data)
    
    Returns:
    - X: Preprocessed features
    - y: Labels
    """
    print(f"Loading dataset from {data_path}...")
    start_time = time.time()
    
    try:
        # Load the dataset, potentially with a smaller sample for testing
        if sample_size:
            print(f"Using sample size of {sample_size} rows for faster processing")
            # Skip rows randomly rather than just taking the first few
            df = pd.read_csv(data_path, low_memory=False, skiprows=lambda i: i>0 and np.random.random() > sample_size/1000000)
        else:
            df = pd.read_csv(data_path, low_memory=False)
        
        print(f"Dataset loaded. Shape: {df.shape}. Time elapsed: {time.time() - start_time:.2f}s")
        
        # Inspect the dataset
        inspect_data(df)
        
        print("\nPreprocessing dataset...")
        
        # Preprocess the dataset
        X, y = preprocess_dataset(df, is_training=True)
        
        print(f"Preprocessing complete. Features shape: {X.shape}. Time elapsed: {time.time() - start_time:.2f}s")
        
        return X, y
    
    except Exception as e:
        print(f"Error loading/preprocessing data: {str(e)}")
        traceback.print_exc()
        raise

def apply_smote(X, y):
    """
    Apply SMOTE to handle class imbalance
    """
    print("Applying SMOTE to handle class imbalance...")
    start_time = time.time()
    
    try:
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"SMOTE applied. New features shape: {X_resampled.shape}. Time elapsed: {time.time() - start_time:.2f}s")
        print(f"Class distribution after SMOTE: {np.bincount(y_resampled.astype(int))}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"Error applying SMOTE: {str(e)}")
        print("Proceeding with original data without SMOTE...")
        return X, y

def train_model(X, y, model_type="xgboost"):
    """
    Train a machine learning model
    
    Parameters:
    - X: Features
    - y: Labels
    - model_type: Type of model to train ("xgboost" or "random_forest")
    
    Returns:
    - Trained model
    """
    print(f"Training {model_type} model...")
    start_time = time.time()
    
    try:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Initialize model based on type
        if model_type.lower() == "xgboost":
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type.lower() == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        print(f"Model training complete. Time elapsed: {time.time() - start_time:.2f}s")
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        # Print evaluation metrics
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return model
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        traceback.print_exc()
        raise

def save_model(model, model_path):
    """
    Save the trained model to disk
    """
    print(f"Saving model to {model_path}...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print("Model saved successfully.")
    
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        traceback.print_exc()

def main():
    """
    Main function to train and save the model
    """
    print("Starting model training pipeline...")
    
    try:
        # For testing with limited resources, use a sample of the data
        # Change this to None to use the full dataset
        sample_size = 10000  # Use 10,000 rows for initial testing
        
        # Load and preprocess the dataset
        X, y = load_and_preprocess_data(DATA_PATH, sample_size=sample_size)
        
        # Check class distribution
        print(f"Class distribution before SMOTE: {np.bincount(y.astype(int))}")
        
        # Apply SMOTE to handle class imbalance
        X_resampled, y_resampled = apply_smote(X, y)
        
        # Train the model
        model = train_model(X_resampled, y_resampled, model_type="xgboost")
        
        # Save the model
        save_model(model, MODEL_PATH)
        
        print("Model training pipeline completed successfully.")
    
    except Exception as e:
        print(f"Error in model training pipeline: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 