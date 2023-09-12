using Plots
using LaTeXStrings

include("../test/test_cld.jl")

# Function to plot the chord length distribution
function plotCLD(cld::Vector)
    histogram(cld, bins=range(0, 6, 100))
    title!("Chord Length Distribution")
    xlabel!("Chord Length")
    ylabel!("Number of Simulated Measurements")
end

function plotCumulCLD(cld::Vector)
    sorted_cld = sort(cld)
    plot(sorted_cld, [i for i = 1:length(sorted_cld)])
    title!("Chord Length Repartition Function")
    xlabel!("Chord Length")
    ylabel!("Cumulative number of measurements")
end

function plotCumulCLD(cld_bins::Vector, bins_number::Vector)
    plot(cld_bins, bins_number)
    title!("Chord Length Repartition Function")
    xlabel!("Chord Length")
    ylabel!("Cumulative number of measurements")
end


# Function to plot the points into a plan
function plot2D(points::Vector, pl::Plan=XY())
    scatter(points, aspect_ratio=:equal)
    title!("Projection ")
    xlabel!(Xlabel(pl))
    ylabel!(Ylabel(pl))
end

plot2D(p::PointsIn2D) = plot2D(vertices(p), plan(p))


# Function to plot the points in the convex hull
function plotConvexHull(points::Vector, convexHullPoints::Vector=convexHull(points), pl::Plan=XY())
    scatter(points, aspect_ratio=:equal, label="all", mc=:blue)
    scatter!(convexHullPoints, aspect_ratio=:equal, label="convex hull", mc=:red)
    xlabel!(Xlabel(pl))
    ylabel!(Ylabel(pl))
    title!("Convex hull of the projection")
end

plotConvexHull(p::PointsIn2D) = plotConvexHull(vertices(p))


# Function to visualise the chord length computed

function plot_chord(points::Vector, _yₗ=missing::Union{Missing,Real})
    ConvV = convexHull(points)
    yₗ = begin
        if ismissing(_yₗ)
            edgeₘᵢₙ, edgeₘₐₓ = [e[2] for e in minAndmax(ConvV, 2)]
            rand() * (edgeₘₐₓ - edgeₘᵢₙ) + edgeₘᵢₙ
        else
            _yₗ
        end
    end
    cl, x_left, x_right = test_chordlength(ConvV, yₗ)
    #@show cl
    scatter(ConvV, aspect_ratio=:equal, label="points", mc=:blue)
    scatter!([(x_left, yₗ), (x_right, yₗ)], aspect_ratio=:equal, label="intersections", mc=:red)
    plot!([(x_left, yₗ), (x_right, yₗ)], aspect_ratio=:equal, lc=:red)
    xlabel!(L"x")
    ylabel!(L"y")
    title!("Chord for a given height")
end

plot_chord(p::PointsIn2D, h::Real) = plot_chord(vertices(p), h::Real)
