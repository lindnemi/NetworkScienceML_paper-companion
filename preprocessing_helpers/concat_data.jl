using Pkg
Pkg.activate(@__DIR__)

using HDF5

#simdir = "data/SN_N020/4Pytorch/"
#results_path = joinpath(@__DIR__, simdir)

results_path = "/p/projects/coen/christian/datasets/snbs_homogeneous_dataset/SN_N100/4Pytorch/"

filename_list = readdir(results_path)
data_names = map(x -> x[1:end-9], filename_list) |> unique
target_names = [tn for tn in data_names if startswith(tn, "s")]

for tn in target_names
    println("Joining", tn)
    tn = endswith(tn, "_id") ? tn[1:end-3] : tn
    key = tn[1:4]
    targets = Vector{Float64}[]

    for fn in [fn for fn in filename_list if startswith(fn,tn)]
        h5open(joinpath(results_path, fn), "r") do file
            push!(targets, read(file)[key])
        end
    end

    targets = reduce(hcat, targets)

    h5open(joinpath(@__DIR__, "grids100",string(tn, "_complete.h5")), "w") do file
        write(file, key, targets)
    end
end
