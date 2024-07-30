import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Load the model and scaler
    model_path = '../models/linkcash_user_identify_model.pkl'
    scaler_path = '../models/scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load and preprocess the data
    file_path = '../data/advanced_synthetic_user_data_test.csv'
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.drop('is_real', axis=1)
    y = data['is_real']
    X_scaled = scaler.transform(X)

    # Evaluate the model
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    report = classification_report(y, y_pred, target_names=['Fake', 'Real'])
    print(report)

    # Feature Importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    feature_importances.plot(kind='bar')
    plt.title('Feature Importance')
    plt.show()

if __name__ == '__main__':
    main()
