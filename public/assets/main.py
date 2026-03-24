import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    config = load_config()
    data_path = config.get('data_path', 'data/raw/dataset.csv')
    target_column = config.get('target_column', 'target')
    
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
    
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    main()