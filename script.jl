include("cld.jl")
include("plot.jl")

using Test

cube = cube(4)

cld = computeCLD(cube)

plotCLD(cld)