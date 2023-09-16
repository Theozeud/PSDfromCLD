include("../scr/plot.jl")

shape = Cube(4)
rot_shape = Rotation(shape, π / 7, π / 2.5, π / 3.3)
edges = [(1,6),(1,7),(1,8),(2,3),(2,4),(2,5),(3,6),(3,8),(4,6),(4,7),(5,7),(5,8)]
plot3D(vertices(rot_shape),edges)