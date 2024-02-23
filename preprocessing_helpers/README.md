# Preprocessing of target values

For every ensemble of power grid topologies the data source provides 10,000 HDF5 files with the target values, one for each grid. In scikit-learn it's easier to work  with a single data vector, which contains the targets for all nodes of all grids. The concat `concat_data.jl` and `concat_mfd.jl` transform the output features accordingly.

# Computation of network measures as input features

Download data, adjust paths, use `make_network_measures_final.ipynb`.