"""Étape 5 : déploiements Prefect (le "worker")."""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    clean_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    train_flow,
)

if __name__ == "__main__":
    deployments = [
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["quality", "mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        clean_flow.to_deployment(
            name="ml-pipeline-clean", tags=["maintenance", "mlops"]
        ),
    ]
    serve(*deployments)