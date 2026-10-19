"""Point d'entrée : appelle les fonctions de model_pipeline.py."""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

MODEL_PATH = "model.joblib"


def main(step):
    X_train, X_test, y_train, y_test = prepare_data()

    if step in ("train", "all"):
        model = train_model(X_train, y_train)
        save_model(model, MODEL_PATH)

    if step in ("evaluate", "all"):
        model = load_model(MODEL_PATH)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projet Churn")
    parser.add_argument("--step", choices=["train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    main(args.step)
