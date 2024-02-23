# Paper companion for `Predicting Instability in Complex Oscillator Networks: Limitations and Potentials of Network Measures and Machine Learning'

The input features and target values are available at:


The power grid topologies needed for computing the network measures are available at:


!Caveat: For this repository to be functional, the absolute and relative paths in the scripts have to be changed.

# Training of models

The models are trained on the two large ensembles and evaluated on the synthetic Texan grid, as well as on four European transmission grids.

Training and evaluation on the synthetic grid happens in the `*.py` files.

`regression.py` and `classifier_for_regression.py` are used for predicting single-node basin stability targets.

`classification.py` and `regressor_for_classification.py` are used for the Troublemaker task.

`tools.py` contains some preprocessing and exporting functions used in the other scripts.

`environment.yaml` contains a conda environment of all packages used for the training.

!Caveat: The memory requirements of models with interactions range up to 64GB RAM

# Outputs for paper

During training a model_info.csv is produced for every model. This contains all relevant meta data, as well as the evaluation results. The real European power grid topolgies were added at a later stage of the project. The notebook `eval_extra_real_grids.ipynb` evaluates all models on the real power grid topolgies, and outputs a single dataframe, which contains the model_info of every single model, extended by the evaluation scores on the real grids.

The notebooks for producing the figures and tables for the paper can be found in `outputs_paper`

`model_info_paper.ipynb` reads that dataframe, concatenates the data on the GNNs used for the paper (given in models_paper), and produces most of the tables and figures shown in the paper.

`corr_extra_real_grids.ipynb` and `corr_mi_paper.ipynb` are used for the Pearson correlation, R^2 and Mutual information plots.

`shap_paper.ipynb` is used for the SHAP value analysis of the gradient boosted trees models.

For the producing the paper outputs a separate conda environment was used. It can be found at `outputs_paper/output_env.yaml`.






