"""Fonctions modulaires pour le projet Churn (prédiction du départ client)."""

import glob

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

TARGET = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


def find_data_file(pattern="Churn_Modelling*"):
    """Trouve le fichier de données (csv ou excel) dans le dossier courant."""
    files = [
        f for f in glob.glob(pattern) if f.lower().endswith((".csv", ".xlsx", ".xls"))
    ]
    if not files:
        raise FileNotFoundError("Fichier de données introuvable dans le dossier.")
    return files[0]


def load_data(path=None):
    path = path or find_data_file()
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def prepare_data(path=None, test_size=0.2, seed=42):
    """Retourne X_train, X_test, y_train, y_test."""
    df = load_data(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_model(X_train, y_train, seed=42):
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model


def save_model(model, path="model.joblib"):
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="model.joblib"):
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    return metrics


def predict(model, X):
    return model.predict(X)
