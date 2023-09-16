include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

shape =  Tetraedron(4)

@time cld = computeCLD(shape, 100000)

plotCLD(cld)
