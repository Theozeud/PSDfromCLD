include("../scr/cld.jl")
using Test

# Test for creating shapes
shape = Cube(4)

@test typeof(shape) <: ConvexPolyhedron
@test length(shape) == 8

# Test for Rotation

rot_shape = Rotation(shape, π, π / 3, π / 4)
#@test #coordonee

# Test for Projection
projected_shape = projectTo(XY(), rot_shape)
# @test

# Test for convexHull

# test on a set of Points
@test Set(convexHull([(4, 1), (1, 3), (2, -4), (2, -2), (1, -2), (1, 1), (-1, 1), (-1, -1), (-2, 2), (-4, -3)])) ==
      Set([(4, 1), (1, 3), (-2, 2), (-4, -3), (2, -4)])


# test for a Cube

convex_shape = convexHull(projected_shape)
# @test

# Test for chordlength

shape = [(0, -1), (-3, 1), (-2, 3), (1, 4)]
above_points = [(-3, 1), (-2, 3), (1, 4)]

shape2 = [(0, 1), (3, -1), (2, -3), (-1, -4)]
below_points = [(3, -1), (2, -3), (-1, -4)]

@test two_points_above(above_points, (0, -1), 2, 1) == [(-3, 1), (1, 4)]
@test two_points_below(below_points, (0, 1), 2, 1) == [(-1, -4), (3, -1)]

carre = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
@test chordlength(carre, 0) == 2

triangle = [(-1, 0), (1, 0), (0, 1)]
@test chordlength(triangle, 0) == 2
@test chordlength(triangle, 1 / 2) == 1


# Performance

