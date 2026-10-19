"""Atelier 3 : pipeline MLOps orchestré avec Prefect."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

SOURCE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
MODEL_PATH = "model.joblib"


def run_cmd(cmd):
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Commande échouée ({result.returncode}) : {' '.join(cmd)}")


@task(name="Installer les dépendances")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run_cmd([sys.executable, "-m", "black", *SOURCE_FILES])


@task(name="Qualité du code")
def check_quality():
    run_cmd([sys.executable, "-m", "flake8", "--max-line-length=100", *SOURCE_FILES])


@task(name="Sécurité du code")
def check_security():
    run_cmd([sys.executable, "-m", "bandit", "-q", "-ll", *SOURCE_FILES])


@task(name="Tests unitaires")
def run_unit_tests():
    test_file = Path("test_model_pipeline.py")
    if not test_file.exists():
        test_file.write_text(
            '"""Exemple de test unitaire."""\n\n'
            "from model_pipeline import prepare_data\n\n\n"
            "def test_prepare_data_returns_four_parts():\n"
            "    result = prepare_data()\n"
            "    assert len(result) == 4\n",
            encoding="utf-8",
        )
    run_cmd([sys.executable, "-m", "pytest", "-v", str(test_file)])


@task(name="Préparation des données")
def prepare_data_task():
    from model_pipeline import prepare_data

    return prepare_data()


@task(name="Entraînement du modèle")
def train_model_task(X_train, y_train):
    from model_pipeline import train_model

    return train_model(X_train, y_train)


@task(name="Sauvegarder le modèle")
def save_model_task(model):
    from model_pipeline import save_model

    save_model(model, MODEL_PATH)
    return MODEL_PATH


@task(name="Charger le modèle")
def load_model_task():
    from model_pipeline import load_model

    return load_model(MODEL_PATH)


@task(name="Évaluer le modèle")
def evaluate_model_task(model, X_test, y_test):
    from model_pipeline import evaluate_model

    return evaluate_model(model, X_test, y_test)


@task(name="Nettoyage")
def clean_task():
    for pattern in ("__pycache__", ".pytest_cache", MODEL_PATH):
        p = Path(pattern)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    return model


@flow(name="evaluate")
def evaluate_flow():
    _, X_test, _, y_test = prepare_data_task()
    model = load_model_task()
    return evaluate_model_task(model, X_test, y_test)


@flow(name="all")
def all_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_unit_tests()
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    return evaluate_model_task(model, X_test, y_test)


@flow(name="clean")
def clean_flow():
    clean_task()


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
    "clean": clean_flow,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
