abstract type Shape end
abstract type 3DShape end <: Shape
abstract type 2DShape end <: Shape

struct Polyhedron <: 3DShape
    vertex::Vector{<:Tuple{<:Real,<:Real,<:Real}}
end

# Polygon
struct Polygon <: 2DShape
    plan::Plan
    vextex::Vector{<:Tuple{<:Real,<:Real}}
end 

#Set of points in 2D (not necessary a plogygon)
struct PointsIn2D <: 2DShape
    plan::Plan
    vextex::Vector{<:Tuple{<:Real,<:Real}}
end


#### Function to create shape in 3D

RectanguarCuboid(a,b,c) = Cuboid([(a/2,b/2,c/2),   (-a/2,-b/2,-c/2), (a/2,-b/2,-c/2), (-a/2,b/2,-c/2),
                      (-a/2,-b/2,c/2), (a/2,b/2,-c/2),   (-a/2,b/2,c/2),  (a/2,-b/2,c/2)])

Cube(c) = RectanguarCuboid(c,c,c)

function Tetraedron(c)
    #c is sidelength
    #coord of the points :
    co = c / (2 * √2) 
    #each couple of points must have opposite values on exactly 2 axis
    #so that the distance is indeed c
    [(co, co, co), (co, -co, -co), (-co, co, -co), (-co, -co, co)]
end

function Pyramide(c, h)
     #c base sidelength and h is height
    [(c / 2, c / 2, 0), (-c / 2, c / 2, 0), (-c / 2, -c / 2, 0), (c / 2, -c / 2, 0), (0, 0, h)]
end