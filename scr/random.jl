
# Known distribution for testing the compution of PSD

normal(r::Real, m::Real, σ::Real) = 1/sqrt(2*σ)*exp(-(r-m)/(2*σ))

uniform(::Real, inf::Real, sup::Real) = 1/(sup-inf)
