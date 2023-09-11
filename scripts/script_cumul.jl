include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")

shape = Tetraedron(4)

#cld = computeCLD(shape, 100000)
#plotCumulCLD(cld)

cld_bins, bins_number = computeCumulCLD(shape, 1000, 10)
plotCumul(cld_bins, bins_number)