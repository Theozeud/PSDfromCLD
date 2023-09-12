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
  
# This function returns a repartition function for the CLD, value is given for each bin
function computeCumulCLD(CLD::Vector, nbins::Int=100)
    max_length = maximum(CLD)
    bins = collect(range(0, max_length, nbins))
    bins_number = zeros(Int,nbins)
    sorted_CLD  = sort(CLD)
    current_cl_index = 1
    for bin_index in 1:nbins
        if bin_index != 1
            bins_number[bin_index] = bins_number[bin_index-1]
        end
        while sorted_CLD[current_cl_index] ≤ bins[bin_index]
            bins_number[bin_index] += 1
            current_cl_index += 1
        end
        bin_index += 1
    end
    bins, bins_number
end
  
function computeCumulCLD(X::Shape3D, ntirage::Int=1, nbins::Int=100, p::Plan=XY(); kwargs...)
    CLD = computeCLD(X, ntirage, p; kwargs...)
    computeCumulCLD(CLD, nbins)
end
  

# Function to compute probability of having a chord length inferior to a given size and with a given computed cumulCLD

function probaCLD(l::Real,cumulcld::Tuple)
    bins, nbins = cumulcld
    index = 0
    for e in bins
        if l≥e
            index += 1
        end
    end
    nbins[index]/ntir
end

function matrixCLD(sizeL::Real, sizeR::Real, L::Vector, R::Vector, cumulcld::Tuple)
    K = zeros(sizeL,sizeR)
    for i in 1:sizeL
        for j in 1:sizeR
            newcumul = (cumulcld[1]*R[j], cumulcld[2])
            K[i,j] = probaCLD(L[i],newcumul)
        end
    end
    K
end