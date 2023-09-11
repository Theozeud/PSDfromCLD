using LinearAlgebra

include("rotation.jl")
include("convexhull.jl")
include("projection.jl")

# Intersection
function intersect2D(p₁, p₂, y)
  @assert length(p₁) == length(p₂) == 2
  (y - p₁[2]) * (p₂[1] - p₁[1]) / (p₂[2] - p₁[2]) + p₁[1]
end

function two_points_above(points, p_below, i, j)
  p1 = _min(points, i)
  theta_max, p2 = 0, points[1]
  for p in points
    theta = _angle(p1, p_below, p)
    if theta > theta_max
      theta_max, p2 = theta, p
    end
  end
  p1[j] < p2[j] ? [p1, p2] : [p2, p1]
end

function two_points_below(points, p_above, i, j)
  p1 = _max(points, i)
  theta_max, p2 = 0, points[1]
  for p in points
    theta = _angle(p1, p_above, p)
    if theta > theta_max
      theta_max, p2 = theta, p
    end
  end
  p1[j] < p2[j] ? [p1, p2] : [p2, p1]
end


function chordlength(points, h)
  above_h = [p for p in points if p[2] > h]
  below_h = [p for p in points if p[2] <= h]

  p_above_min = length(above_h) == 1 ? [_min(above_h, 2)] : two_points_above(above_h, _max(below_h, 2), 2, 1)
  p_below_max = length(below_h) == 1 ? [_max(below_h, 2)] : two_points_below(below_h, _min(above_h, 2), 2, 1)

  x_left = intersect2D(p_above_min[1], p_below_max[1], h)
  x_right = intersect2D(p_above_min[end], p_below_max[end], h)
  (x_right - x_left)
end


# Function to compute one chord length of a shape (randomly if no angle is given)
function computeCL(cp::ConvexPolyhedron, p::Plan=XY(); Φ::Real=2π * rand(), θ::Real=π * rand(), ϕ::Real=π / 2 * rand())
  V = vertices(cp)
  RotV = Rotation(V, Φ, θ, ϕ)
  ProjV = projectTo(p, RotV)
  ConvV = convexHull(ProjV)
  edgeₘᵢₙ, edgeₘₐₓ = [e[index(p)] for e in minAndmax(ConvV, index(p))]
  yₗ = rand() * (edgeₘₐₓ - edgeₘᵢₙ) + edgeₘᵢₙ
  chordlength(ConvV, yₗ)
end


# Function to compute the chord length distribution of a shape
function computeCLD(X::Shape3D, ntirage::Int=1, p::Plan=XY(); kwargs...)
  CLD = zeros(ntirage)
  for i in 1:ntirage
    CLD[i] = computeCL(X, p; kwargs...)
  end
  CLD
end


# Function to compute the matrix K
function matrixCLD(X::Shape3D, nb_l::Int, R::Vector; ntirage::Int = 1, p::Plan=XY(), kwargs...)
  K = zeros(nb_l,length(R))
  base_cld = computeCLD(X, ntirage, p; kwargs...)
  for (ir,r) in zip(R,eachindex(R))
    K[1:end,ir] = base_cld * r
  end
  K
end

