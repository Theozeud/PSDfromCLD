include("cld.jl")
include("plot.jl")


cube = cube(4)

cld = computeCLD(cube)

plotCLD(cld)