module PSDfromCLD
    using LinearAlgebra
    using Distances
    using Optim
    using LaTeXStrings
    using Plots

    export Plan, XY, XZ, YZ
    export Rx, Ry, Rz, Rotation
    export projectTo
    export convexHull
    export computeCL, computeCLD, computeCumulCLD, probaCLD, matrixCLD
    export plotCLD, plotCumulCLD, plot2D, plotConvexHull, plot_chord, plot3D
    export lossCLD, optimizePSD, plotOptiRes, plotError

    include("utils.jl")
    include("random.jl")
    include("plan.jl")
    include("shape.jl")
    include("rotation.jl")
    include("projection.jl")
    include("convexhull.jl")
    include("cld.jl")
    include("plot.jl")
    include("opti.jl")
end