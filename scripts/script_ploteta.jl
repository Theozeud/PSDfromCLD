include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

#Inputs
c = 4

shape = Cube(c)

cld = computeCLD(shape, 100000; Φ = 0, θ = 0)

η₁ = c
η₂ = c*√(2)

plotCLD(cld)

function theory(x)
    if x<c
        return 0
    else
        return c^4/(x^4*sqrt(1-c^2/x^2))
    end
end

plot!(range(0,6,1000),theory.(range(0,6,1000)))
vline!([η₁,η₂])

