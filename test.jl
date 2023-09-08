using Test

# Test for creating shapes

# Test for Rotation

# Test for convexHull

# Test for projection

# Test for chordlength



carre = [(-1,1),(-1,-1),(1,1),(1,-1)]
@test chordlength(carre,0) == 2

triangle = [(-1, 0), (1, 0), (0, 1)]
@test chordlength(triangle, 0) == 2
