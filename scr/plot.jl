using Plots
using LaTeXStrings

include("../test/test_cld.jl")

# Function to plot the chord length distribution
function plotCLD(cld::Vector, nbbins::Int = 100)
    plot(legend = false, grid = false,dpi=300)
    histogram!(cld, bins=range(0, max(cld...), nbbins), normalize=:pdf)
    title!("Chord Length Distribution")
    xlabel!("Chord Length")
    ylabel!("Density")
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
plot2D(p::PointsIn2D) = plot2D(vertices(p), plan(p))

function plot2D(points::Vector, pl::Plan=XY())
    scatter(points, aspect_ratio=:equal)
    title!("Projection ")
    xlabel!(Xlabel(pl))
    ylabel!(Ylabel(pl))
end

# Function to plot the points in the convex hull
plotConvexHull(p::PointsIn2D) = plotConvexHull(vertices(p))

function plotConvexHull(points::Vector, convexHullPoints::Vector=convexHull(points), pl::Plan=XY())
    scatter(points, aspect_ratio=:equal, label="all", mc=:blue)
    scatter!(convexHullPoints, aspect_ratio=:equal, label="convex hull", mc=:red)
    xlabel!(Xlabel(pl))
    ylabel!(Ylabel(pl))
    title!("Convex hull of the projection")
end

# Function to visualise the chord length computed
plot_chord(p::PointsIn2D, h::Real) = plot_chord(vertices(p), h::Real)

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
    _, x_left, x_right = test_chordlength(ConvV, yₗ)
    scatter(ConvV, aspect_ratio=:equal, label="points", mc=:blue, primary=false, grid=false, showaxis=false)
    for i in 1:length(ConvV)-1
        plot!([ConvV[i], ConvV[i+1]], aspect_ratio=:equal, lc=:blue, linewidth=8, dpi=300, primary=false)
    end
    plot!([ConvV[end], ConvV[1]], aspect_ratio=:equal, lc=:blue, linewidth=8, dpi=300, primary=false)
    scatter!([(x_left, yₗ), (x_right, yₗ)], aspect_ratio=:equal, primary=false, mc=:red)
    plot!([(x_left, yₗ), (x_right, yₗ)], aspect_ratio=:equal, lc=:red, linewidth=8, dpi=300, label="Chord", xguidefontsize=20, tickfontsize=5)
    title!("Chord on the projection")
end




# Function to visualise the polyhedron in 2D
plot3D(cp::ConvexPolyhedron) = plot3D(vertices(cp),edges(cp))

function behind(points::Vector,edges::Vector, p::Plan=XY())
    ProjX = projectTo(p, points)
    ConvX = convexHull(ProjX)
    

end

function plot3D(points::Vector, edges::Vector, p::Plan =XY())
    meanz = sum([p[3] for p in points])/length(points)
    behinX = [x for x in points if x[3]<meanz]
    ProjX = projectTo(p, points)
    ConvX = convexHull(ProjX)
    dashX = [setdiff(Set(ProjX),Set(ConvX))...] ∩  projectTo(p, behinX)
    plot(grid=false, showaxis=false)
    for (i,j) in edges
        if ProjX[i] ∈ dashX || ProjX[j] ∈ dashX
            plot!([ProjX[i], ProjX[j]], aspect_ratio=:equal, lc=:blue, linewidth=8, dpi=300, primary=false, ls = :dash)
        else
            plot!([ProjX[i], ProjX[j]], aspect_ratio=:equal, lc=:blue, linewidth=8, dpi=300, primary=false)
        end
    end
    title!("Rotated Cube")
end
