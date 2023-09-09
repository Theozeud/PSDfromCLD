include("shape.jl")
include("cld.jl")
include("plot.jl")
include("test_cld.jl")



shape = Cube(4)

Φ = 3.9420912570528985
ϕ = 0.3783842861610453
θ = 1.3141882213160696
h = -2.881554350650806

rot_shape = Rotation(shape, Φ, θ, ϕ)

#rot_shape = Rotation(shape, π, π / 3, π / 4)
projected_shape = projectTo(XY(), rot_shape)

#plotConvexHull(projected_shape)
plot_chord(projected_shape, h)

#rot_shape = apply(p->Rotation(π,π/3,π/4, p),vertices(shape))

#plot2D(rot_shape)

#plot2D(rot_shape)
#pl = XY()
#npoints = length(rot_shape[1]) == 2 ? rot_shape : apply(p -> projectTo(pl, p), rot_shape)


#@time cld = computeCLD(shape, 10000; Φ=0, ϕ=0)

#plotCLD(cld)

#test_computeCLD(Cube(4), 10)