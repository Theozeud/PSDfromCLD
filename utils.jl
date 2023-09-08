# Utils
_min(points::Vector, i::Int) = points[argmin([p[i] for p in points])]
_max(points::Vector,i ::Int) = points[argmax([p[i] for p in points])]
minAndmax(points::Vector, i::Int) = (_min(points, i),_max(points, i))
apply(f::Base.Callable, X::Vector) = [f(x) for x in X]

# Rotation 

Rx(Φ) = [1   0      0;
         0  cos(Φ) sin(Φ);
         0 -sin(Φ) cos(Φ)]

Ry(θ) = [cos(θ)  0 sin(θ);
         0       1   0;
         -sin(θ) 0 cos(θ)]
       
Rz(ϕ) = [cos(ϕ) sin(ϕ) 0;
         -sin(ϕ) cos(ϕ) 0;
           0     0     1]

Rotation(Φ,θ,ϕ,X) = Tuple(Rx(Φ)*Ry(θ)*Rz(ϕ)*[X...])

# Projection in the plan

abstract type Plan end
struct XY <: Plan end
struct XZ <: Plan end
struct YZ <: Plan end

projectTo(::XY,X) = X[1:2]
projectTo(::XZ,X) = X[1:2:3]
projectTo(::YZ,X) = X[2:3]

index(::XY) = 2
index(::XZ) = 1
index(::YZ) = 3