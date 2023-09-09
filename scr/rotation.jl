# Rotation 

Rx(Φ) = [1 0 0
         0 cos(Φ) sin(Φ)
         0 -sin(Φ) cos(Φ)]

Ry(θ) = [cos(θ) 0 sin(θ)
         0 1 0
         -sin(θ) 0 cos(θ)]

Rz(ϕ) = [cos(ϕ) sin(ϕ) 0
         -sin(ϕ) cos(ϕ) 0
         0 0 1]

Rotation(X::Tuple, Φ, θ, ϕ) = Tuple(Rx(Φ) * Ry(θ) * Rz(ϕ) * [X...])

Rotation(X::Vector, Φ, θ, ϕ) = apply(x -> Rotation(x, Φ, θ, ϕ), X)

Rotation(cp::ConvexPolyhedron, Φ, θ, ϕ) = ConvexPolyhedron(Rotation(vertices(cp), Φ, θ, ϕ))