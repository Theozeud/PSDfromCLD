using Plots

function plotCLD(cld::Vector)
    histogram(cld,bins = range(0,6,100))
    title!("Chord Length Distribution")
end


function plot2D(points::Vector, pl::Plan= XY())
    npoints = length(points[1]) == 2 ? points : apply(p->projectTo(pl,p), points)
    scatter(npoints)
    title!("Projection ")
end