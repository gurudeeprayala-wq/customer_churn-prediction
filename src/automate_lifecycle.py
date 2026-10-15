import json
import sys

import mlflow
import yaml
from mlflow import MlflowClient


def promote_champion():
    with open("config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    registered_model_name = config["registry"]["model_name"]
    decision_metric = config["registry"]["decision_metric"]
    client = MlflowClient()
    versions = list(client.search_model_versions(f"name='{registered_model_name}'"))
    if not versions:
        raise ValueError(f"No versions found for {registered_model_name}")

    candidates = []
    for version in versions:
        run = client.get_run(version.run_id)
        metric_value = run.data.metrics.get(decision_metric)
        if metric_value is not None:
            candidates.append((float(metric_value), int(version.version)))

    if not candidates:
        raise ValueError(f"No registered model version has a {decision_metric} metric")

    champion_value, champion_version = max(candidates)
    client.set_registered_model_alias(
        registered_model_name, "Champion", str(champion_version)
    )
    print(
        f"Champion promoted: {registered_model_name} version {champion_version} "
        f"with {decision_metric} {champion_value:.4f}."
    )


if __name__ == "__main__":
    try:
        promote_champion()
    except (ValueError, mlflow.exceptions.MlflowException) as error:
        print(f"[ERROR] Lifecycle automation failed: {error}")
        sys.exit(1)
