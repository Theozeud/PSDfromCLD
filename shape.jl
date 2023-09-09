
# Different shapes

pave(a,b,c) = [(a/2,b/2,c/2),   (-a/2,-b/2,-c/2), (a/2,-b/2,-c/2), (-a/2,b/2,-c/2),
               (-a/2,-b/2,c/2), (a/2,b/2,-c/2),   (-a/2,b/2,c/2),  (a/2,-b/2,c/2)]

cube(c) = pave(c,c,c)

function tetraedron(c) #c is sidelength
    co = c / (2 * √2) #coord of the points
    return [(co, co, co), (co, -co, -co), (-co, co, -co), (-co, -co, co)]
    #each couple of points must have opposite values on exactly 2 axis
    #so that the distance is indeed c
end

# pyramide


