using Test
include("../scr/cld.jl")

function test_chordlength(points, h)
    above_h = [p for p in points if p[2] > h]
    below_h = [p for p in points if p[2] <= h]

    #p_above_min = length(above_h) == 1 ? [_min(above_h, 2)] : two_min(above_h, 2, 1)
    #p_below_max = length(below_h) == 1 ? [_max(below_h, 2)] : two_max(below_h, 2, 1)

    @show p_above_min = length(above_h) == 1 ? [_min(above_h, 2)] : two_points_above(above_h, _max(below_h, 2), 2, 1)
    @show p_below_max = length(below_h) == 1 ? [_max(below_h, 2)] : two_points_below(below_h, _min(above_h, 2), 2, 1)

    @show h
    @show x_left = intersect2D(p_above_min[1], p_below_max[1], h)
    @show x_right = intersect2D(p_above_min[end], p_below_max[end], h)
    cl = abs(x_right - x_left)


    return cl, x_left, x_right

end

function test_computeCL(cp::ConvexPolyhedron, p::Plan=XY(); Φ::Real=2π * rand(), θ::Real=π * rand(), ϕ::Real=π / 2 * rand())
    @show Φ, ϕ, θ
    V = vertices(cp)
    RotV = Rotation(V, Φ, θ, ϕ)
    ProjV = projectTo(p, RotV)
    ConvV = convexHull(ProjV)
    edgeₘᵢₙ, edgeₘₐₓ = [e[index(p)] for e in minAndmax(ConvV, index(p))]
    yₗ = rand() * (edgeₘₐₓ - edgeₘᵢₙ) + edgeₘᵢₙ
    @show test_chordlength(ConvV, yₗ)
end

function test_computeCLD(X::Shape3D, ntirage::Int=1, p::Plan=XY(); kwargs...)
    CLD = zeros(ntirage)
    for i in 1:ntirage
        CLD[i], _, _ = test_computeCL(X, p; kwargs...)
    end
    CLD
end
