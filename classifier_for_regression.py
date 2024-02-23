## Load modules

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import h5py
import numpy as np
import time

from tools import add_interactions  # custom tools defined in this directory
from tools import save_model
from math import isclose


def run_classification(model, scaler, model_name, with_interactions, random_state):
    
    model = make_pipeline(scaler, VarianceThreshold(),  model)

    task_name = "SNBS"
    model_name = model_name + task_name

    ## Load data sets
    work_dir = "/home/mlindner/coen/micha/netsci_vs_gnn/"

    X20_path = work_dir + "grids20/network_measures_final.csv"
    y20_path = work_dir + "grids20/snbs_complete.h5"
    X100_path = work_dir + "grids100/network_measures_final.csv"
    y100_path = work_dir + "grids100/snbs_complete.h5"
    Xtex_path = work_dir + "gridstexas/network_measures_final.csv"
    ytex_path = work_dir + "gridstexas/snbs_1.h5"

    X20 = pd.read_csv(X20_path).drop(columns=["node_cat", "proper leaf"])
    X100 = pd.read_csv(X100_path).drop(columns=["node_cat", "proper leaf"])
    Xtex = pd.read_csv(Xtex_path).drop(columns=["node_cat", "proper leaf"])

    hf = h5py.File(y20_path, 'r')
    y20 = np.array(hf.get(list(hf.keys())[0])).flatten()
    hf.close()

    hf = h5py.File(y100_path, 'r')
    y100 = np.array(hf.get(list(hf.keys())[0])).flatten()
    hf.close()

    hf = h5py.File(ytex_path, 'r')
    ytex = np.array(hf.get(list(hf.keys())[0])).flatten()
    hf.close()

    ## Add interactions

    if with_interactions:
        X20=X20.drop(
    columns = ['maximal_line_load_post_dc', 'backup_capacity'])
        X100=X100.drop(
    columns = ['maximal_line_load_post_dc', 'backup_capacity'])
        Xtex=Xtex.drop(
    columns = ['maximal_line_load_post_dc', 'backup_capacity'])
        X20=add_interactions(X20)
        X100=add_interactions(X100)
        Xtex=add_interactions(Xtex)


    ## Split test sets

    assert isclose(len(X20) * 0.85 % 1, 0, abs_tol=10e-12)

    X20_test = X20.iloc[int(len(X20) * 0.85):]
    y20_test = y20[int(len(y20) * 0.85):]

    assert isclose(len(X100) * 0.85 % 1, 0, abs_tol=10e-12)

    X100_test = X100.iloc[int(len(X100) * 0.85):]
    y100_test = y100[int(len(y100) * 0.85):]

    ## Loop over both datasets
    for X, y, nodes_per_grid, X_path, y_path in [(X20, y20, 20, X20_path, y20_path),
                                                 (X100, y100, 100, X100_path, y100_path)]:
        # Loop over all train set sizes
        for train_fraction in [0.0007, 0.007, 0.07, 0.7]:

            print("Training", model_name, "on dataset", nodes_per_grid,
                  "with train_fraction", train_fraction, ".")

            ## Train validation split
            assert isclose(len(X) * train_fraction % 1, 0, abs_tol=10e-12)

            X_train = X.iloc[:int(len(X) * train_fraction)]
            y_train = y[:int(len(y) * train_fraction)]
            X_val = None  # We don't do metaparameter studies here
            y_val = None
            
            ## Turn into a classification problem

            Xlog_train = pd.concat([X_train, X_train])
            ylog_train = np.concatenate([np.ones(y_train.size), np.zeros(y_train.size)])
            w = np.concatenate([y_train, 1 - y_train])

            ## Training
            
            tstart = time.time()
            # to pass parameters to a specific step in the pipeline, we need the stepname and a dict of keyword and value.
            # uses string interpolation
            model.fit(Xlog_train, ylog_train, **
                      {f"{type(model[-1]).__name__.lower()}__sample_weight": w})
            ttrain = time.time() - tstart

            sparse_predictors = None
            # if repr(model[-1]).startswith("Orthogonal"):
            #     sparse_predictors = X_train.columns[model[-1].coef_.nonzero()[0]]

            ## Evaluation with r2_score
            # Keep in mind: The model was fit with log loss, not with r2!
            train_r2 = r2_score(
                y_train, model.predict_proba(X_train)[:, 1])
            
            # Test performance
            if nodes_per_grid == 20:
                # For the in-distribution test, scale the data with the train set
                ev20 = r2_score(y20_test, model.predict_proba(
                    X20_test)[:, 1])
                # For out-of-distribution, scale with all grids of that size to correct size effects
                model[0].fit(X100)
                ev100 = r2_score(y100_test, model.predict_proba(
                    X100_test)[:, 1])
            if nodes_per_grid == 100:
                # For the in-distribution test, scale the data with the train set
                ev100 = r2_score(y100_test, model.predict_proba(
                    X100_test)[:, 1])
                # For out-of-distribution, scale with all grids of that size to correct size effects
                model[0].fit(X20)
                ev20 = r2_score(y20_test, model.predict_proba(
                    X20_test)[:, 1])
                
            # Texas is always an out-of-distribution test
            model[0].fit(Xtex)                
            evTex = r2_score(ytex, model.predict_proba(Xtex)[:, 1])

            print("Train R2:", train_r2)
            print("ev20 R2:", ev20)
            print("ev100 R2:", ev100)
            print("evTex R2:", evTex)

            # Model persistence
            ## Save the model for later evaluation.

            save_model(model_name, model, scaler, task_name, random_state, nodes_per_grid, work_dir, X_path, y_path, X_train.index, None, X20_test.index,
                       X100_test.index, ttrain, with_interactions, train_r2=train_r2,
                       ev20_r2=ev20, ev100_r2=ev100, evTex_r2=evTex, sparse_predictors=sparse_predictors)



if __name__ == "__main__":
    ## Check for correct conda environment
    import sys
    assert sys.path[2].startswith("/p/projects/coen/micha/netsci_vs_gnn/envs/")

    ## Import Models
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    ## Run Models

    for random_state in [1, 2, 3, 4, 5]:
        run_classification(
            model=HistGradientBoostingClassifier(
                max_iter=1750,
                early_stopping=False,
                random_state=random_state),
            scaler=StandardScaler(),
            model_name="GradientBoostingClassifier",
            with_interactions=False,
            random_state=random_state)
        
    run_classification(
        model=LogisticRegression(max_iter=1000000, penalty=None),
        scaler=StandardScaler(),
        model_name="LogisticRegression",
        with_interactions=True,
        random_state=None)
