using Plots
using LaTeXStrings

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
    @show points
    @show convexHullPoints
    scatter(points, aspect_ratio=:equal, label="all", mc=:blue)
    scatter!(convexHullPoints, aspect_ratio=:equal, label="convex hull", mc=:red)
    xlabel!(L"x")
    ylabel!(L"y")
    title!("Convex hull of the projection")

end

plotConvexHull(p::PointsIn2D) = plotConvexHull(vertices(p))