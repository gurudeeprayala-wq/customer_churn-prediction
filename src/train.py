import numpy as np
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier

def run_training():
    print("Starting Model Training...")
    with open('config.yaml', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    processed_dir = config['data']['processed_dir']
    
    # 1. Load processed training data
    X_train_final = np.load(f"{processed_dir}/X_train_final.npy")
    y_train = np.load(f"{processed_dir}/y_train.npy")
    
    # 2. Initialize and train the model
    model = RandomForestClassifier(**config['model'])
    
    model.fit(X_train_final, y_train)
    
    # 3. Save the trained model
    model_path = config['paths']['model']
    joblib.dump(model, model_path)
    print("Model training complete and saved to disk!")

if __name__ == "__main__":
    run_training()