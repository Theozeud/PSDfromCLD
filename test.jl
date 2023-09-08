using Test

# Test for creating shapes
shape = cube(4)

@test length(shape) == 8

rot_shape = apply(p->Rotation(π,π/3,π/4, p),shape)
plot2D(rot_shape)


# Test for Rotation

# Test for convexHull

# Test for projection

# Test for chordlength



carre = [(-1,1),(-1,-1),(1,1),(1,-1)]
@test chordlength(carre,0) == 2

triangle = [(-1, 0), (1, 0), (0, 1)]
@test chordlength(triangle, 0) == 2


# Performance

