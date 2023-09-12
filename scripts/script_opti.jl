include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

include("../scr/opti.jl")
include("../scr/random.jl")


# Inputs
sizeR = 3
sizeL = 3
ntir = 1000000

# Random size
random = Uniform(10,5)
R = tir(random, sizeR)

# Given L (comes from the mesurement)
L = range(1,4,sizeL)

# Base Shape
shape = Cube(1)

# Utils function
bins, bins_number = computeCumulCLD(shape, ntir, 10000)

# Computation of the matrix K
K = matrixCLD()

# Computation of the CLD for different size
function computeCLDdifferentSize()
    cld = zeros(ntir)
    for i in 1:ntir
        r = tir(random, 1)[1]
        cp = Cube(r)
        cld[i] = computeCL(cp)
    end
    bins,bins_number = computeCumulCLD(cld, 10000)
    [probaCLD(l,1,bins,bins_number) for l in L]
end
q = computeCLDdifferentSize()


# Optimisation
ψ₀ = zeros(sizeR)
method = LBFGS()
ε = 0.1

opt = optimizePSD(K, q, ε, method, ψ₀)

trueresult = [law(random,r) for r in R]

res = Optim.minimizer(opt)

