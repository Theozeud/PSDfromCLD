include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")



shape = RectanguarCuboid(2, 4, 6)


@time cld = computeCLD(shape, 500000)

plotCLD(cld)