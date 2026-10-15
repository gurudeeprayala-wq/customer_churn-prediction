import os
import json
import joblib
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer

def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    with open('config.yaml', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    data_config = config['data']
    split_config = config['split']
    processed_dir = data_config['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Load the raw data
    data_path = data_config['raw_path']
    df = pd.read_csv(data_path)
    
    # 2. Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # 2. Data Cleaning (Experiment: Fill missing TotalCharges with KNN Imputer)
    # First, force empty strings to NaN
    # df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # # We use tenure and MonthlyCharges to help KNN find the most similar customers
    # knn_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # print("Applying KNN Imputation")
    # imputer = KNNImputer(n_neighbors=5)
    # df[knn_cols] = imputer.fit_transform(df[knn_cols])

    id_column = data_config['id_column']
    target_column = data_config['target_column']
    if id_column in df.columns:
        df = df.drop(id_column, axis=1)
        
    # Unconditional string to binary integer conversion
    df[target_column] = df[target_column].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)
        
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(split_config['test_size']),
        random_state=int(split_config['random_state']),
        stratify=y,
    )
    
    # 4. Separate Column Types
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # 5. Scale & Encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])
    
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])
    
    # Combine Features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))
    
    # 6. Save Artifacts with explicit integer type casting
    np.save(os.path.join(processed_dir, 'X_train_final.npy'), X_train_final)
    np.save(os.path.join(processed_dir, 'X_test_final.npy'), X_test_final)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train.to_numpy(dtype=np.int64))
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test.to_numpy(dtype=np.int64))
    
    joblib.dump(scaler, config['paths']['scaler'])
    joblib.dump(ohe, config['paths']['encoder'])
    joblib.dump(
        {'scaler': scaler, 'encoder': ohe},
        config['paths']['preprocessor'],
    )
    
    # Save Metadata
    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols)
    }
    with open(os.path.join(processed_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()