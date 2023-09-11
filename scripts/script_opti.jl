include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

include("../scr/opti.jl")
include("../scr/random.jl")


# Inputs
sizeR = 2
sizeL = 2
ntir = 10

# Random size
random = Normal(3,1)
R = tir(random, sizeR)

# Given L
L = range(1,4,sizeL)

# Base Shape
shape = Cube(1)

# Utils function
bins, bins_number = computeCumulCLD(shape, ntir, 10000)

function probaCLD(l,r,bins,bins_number)
    index = 0
    for e in bins
        if l≥e*r
            index += 1
        end
    end
    bins_number[index]/ntir
end

# Computation of the matrix K
function matrixCLD()
    K = zeros(sizeR,sizeL)
    for i in 1:sizeR
        for j in 1:sizeL
            K[i,j] = probaCLD(L[j],R[i],bins,bins_number)
        end
    end
    K
end
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
ψ₀ = 0
method = LBFGS()
ε = 0.1

optimizePSD(K, q, ε, method, ψ₀)