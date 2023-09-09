include("projection.jl")

abstract type AbstractShape end

abstract type Shape3D <: AbstractShape end
abstract type Shape2D <: AbstractShape end

struct ConvexPolyhedron <: Shape3D
    vertices::Vector{<:Tuple{<:Real,<:Real,<:Real}}
end

@inline vertices(p::ConvexPolyhedron) = p.vertices

# Polygon
struct ConvexPolygon <: Shape3D
    plan::Plan
    vertices::Vector{<:Tuple{<:Real,<:Real}}
end 

@inline plan(p::ConvexPolygon) = p.plan
@inline vertices(p::ConvexPolygon) = p.vertices

#Set of points in 2D (not necessary a plogygon)
struct PointsIn2D <: Shape2D
    plan::Plan
    vertices::Vector{<:Tuple{<:Real,<:Real}}
end

@inline plan(p::PointsIn2D) = p.plan
@inline vertices(p::PointsIn2D) = p.vertices


#### Function to create shape in 3D

RectanguarCuboid(a,b,c) = ConvexPolyhedron([(a/2,b/2,c/2),   (-a/2,-b/2,-c/2), (a/2,-b/2,-c/2), (-a/2,b/2,-c/2),
                      (-a/2,-b/2,c/2), (a/2,b/2,-c/2),   (-a/2,b/2,c/2),  (a/2,-b/2,c/2)])

Cube(c) = RectanguarCuboid(c,c,c)

function Tetraedron(c)
    #c is sidelength
    #coord of the points 
    co = c / (2 * √2) 
    #each couple of points must have opposite values on exactly 2 axis
    #so that the distance is indeed c
    ConvexPolyhedron([(co, co, co), (co, -co, -co), (-co, co, -co), (-co, -co, co)])
end

function Pyramide(c, h)
     #c base sidelength and h is height
     ConvexPolyhedron([(c / 2, c / 2, 0), (-c / 2, c / 2, 0), (-c / 2, -c / 2, 0), (c / 2, -c / 2, 0), (0, 0, h)])
end