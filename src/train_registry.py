import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score


def train_and_register():
    print("Starting model training and registration...")
    with open("config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    processed_dir = config["data"]["processed_dir"]
    X_train = np.load(f"{processed_dir}/X_train_final.npy")
    X_test = np.load(f"{processed_dir}/X_test_final.npy")
    y_train = np.load(f"{processed_dir}/y_train.npy")
    y_test = np.load(f"{processed_dir}/y_test.npy")

    params = config["model"]
    experiment_name = config["mlflow"]["experiment_name"]
    registered_model_name = config["registry"]["model_name"]
    decision_metric = config["registry"]["decision_metric"]
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_Registry") as run:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        recall = recall_score(y_test, model.predict(X_test), zero_division=0)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_params(params)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_param("model_family", "RandomForest")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    registered_model = mlflow.register_model(model_uri, registered_model_name)
    print(
        f"Registered {registered_model_name} version {registered_model.version} "
        f"with {decision_metric}={recall if decision_metric == 'recall' else roc_auc:.4f}."
    )


if __name__ == "__main__":
    train_and_register()
