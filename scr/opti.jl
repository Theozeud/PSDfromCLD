using Distances

function loss(K::Matrix, ψ::Vector, q::Vector, ε::Real)
    @assert axes(K)[2] == length(ψ) == length(q)
    sqeuclidean(K*ψ,q)+ ε*norm(ψ)^2
end

