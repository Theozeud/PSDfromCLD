include("shape.jl")
include("cld.jl")
include("plot.jl")




shape = Cube(4)
rot_shape = Rotation(shape, π/8, π / 7, 7π /8)
projected_shape = projectTo(XY(), rot_shape)

#plotConvexHull(projected_shape)
#plot_chord(projected_shape)


#@time cld = computeCLD(shape, 10000; Φ=0, ϕ=0)

#plotCLD(cld)
