include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")



shape = Cube(4)
#shape =  Tetraedron(4)


#Φ = 3.9420912570528985
#ϕ = 0.3783842861610453
#θ = 1.3141882213160696
#h = -2.881554350650806



@time cld = computeCLD(shape, 100000)

plotCLD(cld)
