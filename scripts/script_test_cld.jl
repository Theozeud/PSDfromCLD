include("../scr/plot.jl")

shape = Cube(4)
h = 1.5
rot_shape = Rotation(shape, π / 7, π / 2.5, π / 3.3)
projected_shape = projectTo(XY(), rot_shape)

plot_chord(projected_shape, h)