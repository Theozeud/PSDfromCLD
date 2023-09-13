include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

include("../scr/opti.jl")
include("../scr/random.jl")


# Inputs
sizeR = 100
sizeL = 100
ntir = 1000000
ntirCLD = ntir

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
rList = collect(range(0, rMax, sizeR))
lList = collect(range(0, lMax, sizeL))
cumulCLD = computeCumulCLD(Ltir)
K = matrixCLD(sizeL, sizeR, lList, rList, cumulCLD)

# Computation of the CLD for different size
#function computeCLDdifferentSize()
#    cld = zeros(ntir)
# Random measurement
#    for i in 1:ntir
#        r = tir(random, 1)[1]
#        cp = Cube(r)
#        cld[i] = computeCL(cp)
#    end
#    bins, bins_number = computeCumulCLD(cld, 10000)
#    [probaCLD(l, cumulCLD) for l in range(0, lMax, sizeL)]
#end

# Computation of CLD with q = KΨ
Ψ = apply(x -> law(random, x), rList) # Exact distribution
q = K * Ψ

# Optimisation
ψ₀ = [0.2 + rand() * 0.01 for i in 1:sizeR] # Initial distribution

method = LBFGS()
ε = 0.1 # Regularisation coefficient

opt = optimizePSD(K, q, ε, method, ψ₀)

res = Optim.minimizer(opt)

plotOptiRes(rList, res, Ψ, ψ₀)