include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

shape = Cube(4)

@time cld = computeCLD(shape, 1000000)

plotCLD(cld,5000)
