using Plots
using LaTeXStrings
include("test_cld.jl")

function plotCLD(cld::Vector)
    histogram(cld, bins=range(0, 6, 100))
    title!("Chord Length Distribution")
    xlabel!("Chord Length")
    ylabel!("Number of Simulated Measurements")

end


function plot2D(points::Vector, pl::Plan=XY())
    npoints = length(points[1]) == 2 ? points : apply(p -> projectTo(pl, p), points)
    scatter(npoints, aspect_ratio=:equal)
    title!("Projection ")
    xlabel!(L"x")
    ylabel!(L"y")
end

plot2D(p::PointsIn2D) = plot2D(vertices(p), plan(p))

function plotConvexHull(points::Vector, convexHullPoints::Vector=convexHull(points))
    scatter(points, aspect_ratio=:equal, label="all", mc=:blue)
    scatter!(convexHullPoints, aspect_ratio=:equal, label="convex hull", mc=:red)
    xlabel!(L"x")
    ylabel!(L"y")
    title!("Convex hull of the projection")

end

plotConvexHull(p::PointsIn2D) = plotConvexHull(vertices(p))

function plot_chord(points::Vector)
    p = XY()
    ConvV = convexHull(points)
    edgeₘᵢₙ, edgeₘₐₓ = [e[index(p)] for e in minAndmax(ConvV, index(p))]
    yₗ = rand() * (edgeₘₐₓ - edgeₘᵢₙ) + edgeₘᵢₙ
    cl, x_left, x_right = test_chordlength(ConvV, yₗ)
    @show cl
    scatter(ConvV, aspect_ratio=:equal, label="points", mc=:blue)
    scatter!([(x_left, yₗ), (x_right, yₗ)], aspect_ratio=:equal, label="intersections", mc=:red)
    plot!([(x_left, yₗ), (x_right, yₗ)], aspect_ratio=:equal, lc=:red)
    xlabel!(L"x")
    ylabel!(L"y")
    title!("Chord for a given height")

end

plot_chord(p::PointsIn2D) = plot_chord(vertices(p))
