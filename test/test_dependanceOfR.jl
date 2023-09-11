include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")


function plotCLDwrtR(nb_r::Int)
    result = Real[]
    shape = Cube(1)
    cld_i = computeCLD(shape, 10000)
    for r in range(1,4,nb_r)
        shape = Cube(r)
        cld = computeCLD(shape, 10000)
        push!(result,sum(cld)/sum(cld_i))
    end
    plot(range(1,4,nb_r), result)
end

plotCLDwrtR(50)