include("cld.jl")
include("plot.jl")
include("shape.jl")



shape = cube(4)

@time calcvul = computeCLD(shape, 100000; Φ=0, ϕ=0)

plotCLD(calcvul)