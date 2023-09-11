include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

include("../scr/opti.jl")
include("../scr/random.jl")


shape = Cube(1)

K = matrixCLD(shape, 3, [1,2,3]; ntirage = 3)
q = 

method = LBFGS()
ψ₀ = 0

optimizePSD(K, q, ε, method, ψ₀)