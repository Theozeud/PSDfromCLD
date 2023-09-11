
# Known distribution for testing the compution of PSD

normal(r::Real, m::Real, σ::Real) = 1/sqrt(2*σ)*exp(-(r-m)/(2*σ))

uniform(::Real, a::inf, b::sup) = 1/(b-a)
