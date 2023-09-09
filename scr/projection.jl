
include("plan.jl")
include("shape.jl")

# Projection for a point
projectTo(::XY, X::Tuple) = X[1:2]
projectTo(::XZ, X::Tuple) = X[1:2:3]
projectTo(::YZ, X::Tuple) = X[2:3]

# Projection for a Vector of points
projectTo(p::Plan, X::Vector) = apply(x -> projectTo(p, x), X)

# Projection for a ConvexPolyhedron
projectTo(p::Plan, cp::ConvexPolyhedron) = PointsIn2D(p, projectTo(p, vertices(cp)))

