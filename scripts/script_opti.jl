include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

include("../scr/opti.jl")
include("../scr/random.jl")


# Inputs
sizeR = 100
sizeL = 100
ntir = 1000000
Aₙₒᵢₛₑ = 0 # noise added to q 

# Choice of the Particle Sized Distribution
#random = Uniform(5, 10)
random = Normal(5, 0.5)

# Base Shape
shape = Cube(1)

# Ltir is randomly chosen in the CLD obtained with the shape (measurement)
Ltir = computeCLD(shape, ntir)

# Computation of the matrix K
rMax = 10 # max particle size
lMax = rMax * maximum(Ltir) # max chord length
rVector = collect(range(0, rMax, sizeR))
lVector = collect(range(0, lMax, sizeL))
cumulCLD = computeCumulCLD(Ltir)
K = matrixCLD(lVector, rVector, cumulCLD)

# Computation of CLD with q = KΨ
ψ = apply(x -> law(random, x), rVector) # Exact distribution
q = K * ψ + [rand() * Aₙₒᵢₛₑ for i in 1:sizeL]

# Optimisation
ψ₀ = [0.2 + rand() * 0.01 for i in 1:sizeR] # Initial distribution
ε = 1 # Regularisation coefficient
ψₘ = 0

opt = optimizePSD(K, q, ε, ψₘ, ψ₀)

res = Optim.minimizer(opt)

plotOptiRes(rVector, res, ψ, ψ₀)