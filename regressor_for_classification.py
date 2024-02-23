## Load modules

from sklearn.metrics import precision_recall_fscore_support
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


def run_regression(model, scaler, model_name, with_interactions, random_state):
    
    model = make_pipeline(scaler, VarianceThreshold(),  model)
    
    task_name = "Troublemaker"
    model_name = model_name + task_name

    ## Load data sets
    work_dir = "/home/mlindner/coen/micha/netsci_vs_gnn/"

    X20_path = work_dir + "grids20/network_measures_final.csv"
    y20_path = work_dir + "grids20/max_freq_dev.h5"
    X100_path = work_dir + "grids100/network_measures_final.csv"
    y100_path = work_dir + "grids100/max_freq_dev.h5"
    Xtex_path = work_dir + "gridstexas/network_measures_final.csv"
    ytex_path = work_dir + "gridstexas/max_freq_dev.h5"

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
        X20 = X20.drop(
            columns=['maximal_line_load_post_dc', 'backup_capacity'])
        X100 = X100.drop(
            columns=['maximal_line_load_post_dc', 'backup_capacity'])
        Xtex = Xtex.drop(
            columns=['maximal_line_load_post_dc', 'backup_capacity'])
        X20 = add_interactions(X20)
        X100 = add_interactions(X100)
        Xtex = add_interactions(Xtex)

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

            ## Fit survivability


            tstart = time.time()
            model.fit(X_train, y_train)
            ttrain = time.time() - tstart
            
            sparse_predictors = None
            if repr(model[-1]).startswith("Orthogonal"):
                sparse_predictors = X_train.columns[model[-1].coef_.nonzero()[0]]

            ## Train performance
            train_r2 = r2_score(y_train, model.predict(X_train))
            
            # Threshold at 1 to classify nodes as troublemakers
            train_score = precision_recall_fscore_support(
                y_train > 15.0,
                model.predict(X_train) > 15.0,
                beta=2)[1:3]
            # This extracts the scores of the True class
            train_score = list(map(lambda x: x[1], train_score))
            
            # Test performance
            if nodes_per_grid == 20:
                # For the in-distribution test, scale the data with the train set
                ev20_r2 = r2_score(y20_test, model.predict(X20_test))
                ev20_score = precision_recall_fscore_support(
                    y20_test > 15.0,
                    model.predict(X20_test) > 15.0,
                    beta=2)[1:3]
                ev20_score = list(map(lambda x: x[1], ev20_score))
                 
                # For out-of-distribution, scale with all grids of that size to correct size effects
                model[0].fit(X100)
                ev100_r2 = r2_score(y100_test, model.predict(X100_test))
                ev100_score = precision_recall_fscore_support(
                    y100_test > 15.0,
                    model.predict(X100_test) > 15.0,
                    beta=2)[1:3]
                ev100_score = list(map(lambda x: x[1], ev100_score))
                
            if nodes_per_grid == 100:
                # in distribution
                ev100_r2 = r2_score(y100_test, model.predict(X100_test))
                ev100_score = precision_recall_fscore_support(
                    y100_test > 15.0,
                    model.predict(X100_test) > 15.0,
                    beta=2)[1:3]
                ev100_score = list(map(lambda x: x[1], ev100_score))
                
                # out of distribution
                model[0].fit(X20)
                ev20_r2 = r2_score(y20_test, model.predict(X20_test))
                ev20_score = precision_recall_fscore_support(
                    y20_test > 15.0,
                    model.predict(X20_test) > 15.0,
                    beta=2)[1:3]
                ev20_score = list(map(lambda x: x[1], ev20_score))
                
            # Texas is always an out-of-distribution test
            model[0].fit(Xtex)
            evTex_r2 = r2_score(ytex, model.predict(Xtex))
            evTex_score = precision_recall_fscore_support(
                ytex > 15.0,
                model.predict(Xtex) > 15.0,
                beta=2)[1:3]
            evTex_score = list(map(lambda x: x[1], evTex_score))

            print("Train R2:", train_r2)
            print("ev20 R2:", ev20_r2)
            print("ev100 R2:", ev100_r2)
            print("evTex R2:", evTex_r2)
            

            print("Train Recall:", train_score[0], "Train F2:", train_score[1])
            print("ev20 Recall:", ev20_score[0], "ev20 F2:", ev20_score[1])
            print("ev100 Recall:", ev100_score[0], "ev100 F2:", ev100_score[1])
            print("evTex Recall:", evTex_score[0], "evTex F2:", evTex_score[1])
            
            

            # Model persistence
            ## Save the model for later evaluation.

            save_model(model_name, model, scaler, task_name, random_state, nodes_per_grid, work_dir, X_path, y_path, X_train.index, None, X20_test.index,
                       X100_test.index, ttrain, with_interactions, train_r2=train_r2,
                       ev20_r2=ev20_r2, ev100_r2=ev100_r2, evTex_r2=evTex_r2, train_recall=train_score[0],
                       ev20_recall=ev20_score[0], ev100_recall=ev100_score[0], evTex_recall=evTex_score[0], train_f2=train_score[1],
                       ev20_f2=ev20_score[1], ev100_f2=ev100_score[1], evTex_f2=evTex_score[1], sparse_predictors=sparse_predictors)


def f2score(precision, recall):
    return ((5 * precision * recall) / (4 * precision + recall))

if __name__ == "__main__":
    ## Check for correct conda environment
    import sys
    assert sys.path[2].startswith("/p/projects/coen/micha/netsci_vs_gnn/envs/")

    ## Import Models
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import OrthogonalMatchingPursuit

    ## Run Models

    run_regression(
        model=OrthogonalMatchingPursuit(n_nonzero_coefs=1),
        scaler=StandardScaler(),
        model_name="omp1Linear",
        with_interactions=True,
        random_state=None)

    run_regression(
        model=OrthogonalMatchingPursuit(n_nonzero_coefs=2),
        scaler=StandardScaler(),
        model_name="omp2Linear",
        with_interactions=True,
        random_state=None)

    run_regression(
        model=OrthogonalMatchingPursuit(n_nonzero_coefs=5),
        scaler=StandardScaler(),
        model_name="omp5Linear",
        with_interactions=True,
        random_state=None)

    for random_state in [1, 2, 3, 4, 5]:
        run_regression(
            model=HistGradientBoostingRegressor(
                max_iter=1750,
                early_stopping=False,
                random_state=random_state),
            scaler=StandardScaler(),
            model_name="GradientBoostingRegression",
            with_interactions=False,
            random_state=random_state)

    run_regression(
        model=LinearRegression(),
        scaler=StandardScaler(),
        model_name="LinearRegression",
        with_interactions=True,
        random_state=None)
    
