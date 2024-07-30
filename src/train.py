import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from tqdm import tqdm
import time

def update_progress_bar(progress_bar, desc):
    progress_bar.set_description(desc)
    progress_bar.update(1)
    time.sleep(0.1)

def main():
    total_steps = 7
    progress_bar = tqdm(total=total_steps)

    # Load and preprocess the data
    update_progress_bar(progress_bar, "Loading and preprocessing data")
    file_path = '../data/advanced_synthetic_user_data_test.csv'
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.drop('is_real', axis=1)
    y = data['is_real']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    update_progress_bar(progress_bar, "Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    update_progress_bar(progress_bar, "Training the model")
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and scaler
    update_progress_bar(progress_bar, "Saving the trained model")
    joblib.dump(model, '../models/linkcash_user_identify_large_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')

    # Close the progress bar
    progress_bar.close()

if __name__ == '__main__':
    main()
