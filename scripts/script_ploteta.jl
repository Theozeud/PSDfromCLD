include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")



shape =  ConvexPolyhedron([(-2,0,0),(2,0,0),(-1,1,0),(1,1,0)])

cld = computeCLD(shape, 1000000; Φ = 0, θ = 0)

plotCLD(cld)

#=
function theory(x)
    if x<c
        return 0
    else
        return c^4/(x^4*sqrt(1-c^2/x^2))
    end
end

plot!(range(0,6,1000),theory.(range(0,6,1000)))

=#