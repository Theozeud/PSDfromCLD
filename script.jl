include("cld.jl")
include("plot.jl")
include("shape.jl")


using Test

cube = cube(4)

cld = computeCLD(cube)

plotCLD(cld)