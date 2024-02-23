using HDF5

targets = Vector{Float64}[]

#for _path in ["train", "valid", "test"]
    path = joinpath("/p/projects/coen/christian/datasets/snbs_homogeneous_dataset/texas/dataset_texas/")#, _path)
    for fn in [fn for fn in readdir(path) if startswith(fn,
        "trouble_maker_init_threshold_2-5percentile_name_100_id")]
        h5open(joinpath(path, fn), "r") do file
            push!(targets, read(file)["nodal_max_freq_dev"])
        end
    end
#end

targets = reduce(hcat, targets)

h5open(joinpath("/p/projects/coen/micha/netsci_vs_gnn", "gridstexas", string("max_freq_dev.h5")), "w") do file
    write(file, "max_freq_dev", targets)
end
