
# Projection in the plan
abstract type Plan end
struct XY <: Plan end
struct XZ <: Plan end
struct YZ <: Plan end


index(::XZ) = 1
index(::XY) = 2
index(::YZ) = 3

Xlabel(::XY) = L"x"
Xlabel(::YZ) = L"y"
Xlabel(::XZ) = L"z"
Ylabel(::XY) = L"y"
Ylabel(::YZ) = L"z"
Ylabel(::XZ) = L"z"

# Projection for a point
projectTo(::XY, X::Tuple) = X[1:2]
projectTo(::XZ, X::Tuple) = X[1:2:3]
projectTo(::YZ, X::Tuple) = X[2:3]

# Projection for a Vector of points
projectTo(p::Plan, X::Vector) = apply(x -> projectTo(p, x), X)



