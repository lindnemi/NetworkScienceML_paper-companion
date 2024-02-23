from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def add_interactions(X, drop = True, include_bias=False, verbose=False):
    poly = PolynomialFeatures(2, include_bias=include_bias) # by default no constant term
    X = pd.DataFrame(
        poly.fit_transform(X),
        columns=poly.get_feature_names_out(X.columns))

    if drop:
        dup_cols = ["bulk^2", "dense sprout^2",
                    "inner tree node^2",  "root^2", "sparse sprout^2", "node_connected_to_max_load_line^2", "connected_post^2"]
        if verbose:
            print("Dropping duplicate interaction terms:", dup_cols)
        X = X.drop(columns=dup_cols)
    return X


from joblib import dump
from pathlib import Path
from datetime import datetime
import git
import subprocess
import socket


def save_model(model_name, model, scaler, task_name, random_state, nodes_per_grid, work_dir,
               X_path, y_path, train_idx, val_idx, test20_idx, test100_idx, ttrain,
               with_interactions,  verbose=False,
               train_r2=None, ev20_r2=None, ev100_r2=None, evTex_r2=None,
               train_recall=None, ev20_recall=None, ev100_recall=None, evTex_recall=None,
               train_f2=None, ev20_f2=None, ev100_f2=None, evTex_f2=None, sparse_predictors=None):
    run = 1
    while True:
        model_path = work_dir + "models_paper/" + model_name + f"/{run:03}"
        try:
            Path(model_path).mkdir(parents=True)
        except FileExistsError:
            if verbose:
                print(f"Run {run:03} already exists.")
            run += 1
        else:
            break

    model_path += "/"

    dump(model,  model_path + "model.joblib")

    ## Save model performance and meta information as well.

    repo = git.Repo(".")

    model_info = {"model_name": model_name,
                  "model": repr(model),
                  "scaler": repr(scaler),
                  "task": task_name,
                  "random_state": random_state,
                  "datetime": str(datetime.now()),
                  "nodes_per_grid": nodes_per_grid,
                  "features_path": X_path,
                  "labels_path": y_path,
                  "train_r2": train_r2,
                  "ev20_r2": ev20_r2,
                  "ev100_r2": ev100_r2,
                  "evTex_r2": evTex_r2,
                  "train_f2": train_f2,
                  "ev20_f2": ev20_f2,
                  "ev100_f2": ev100_f2,
                  "evTex_f2": evTex_f2,
                  "train_recall": train_recall,
                  "ev20_recall": ev20_recall,
                  "ev100_recall": ev100_recall,
                  "evTex_recall": evTex_recall,
                  "sparse_predictors": sparse_predictors,
                  "training_time": ttrain,
                  "with_interactions": with_interactions,
                  "train_idx": train_idx,
                  "num_grids_for_training": len(train_idx) / nodes_per_grid,
                  "val_idx": val_idx, 
                  "test20_idx": test20_idx,
                  "test100_idx": test100_idx,
                  "gitsha": repo.head.object.hexsha,
                  "gitdiff": repo.git.diff(),
                  "gitbranch": repo.active_branch.name,
                  "gitremote": repo.remotes.origin.url,
                  "hostname": socket.gethostname(),
                  "model_path": model_path,
                  "script_path": __file__}
    
    pd.DataFrame.from_dict([model_info]).to_csv(
        model_path + "model_info.csv", index=False, header=True)

    ## Copy the used conda environment to the folder for reproducibility. In this case `./envs`.

    shell_command = "conda env export"

    with open(model_path + "environment.yml", "w") as f:
        subprocess.run(shell_command.split(), stdout=f)
