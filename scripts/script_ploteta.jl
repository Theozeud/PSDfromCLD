include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

#Inputs
c = 4

shape = Cube(c)

cld = computeCLD(shape, 100000)

η₁ = c
η₂ = c*√(2)

plotCLD(cld)
vline!([η₁,η₂])

