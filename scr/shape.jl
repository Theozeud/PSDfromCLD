include("plan.jl")

abstract type AbstractShape end

abstract type Shape3D <: AbstractShape end

struct ConvexPolyhedron <: Shape3D
    vertices::Vector{<:Tuple{<:Real,<:Real,<:Real}}
end

@inline vertices(p::ConvexPolyhedron) = p.vertices
Base.length(p::ConvexPolyhedron) = length(vertices(p))

abstract type Shape2D <: AbstractShape end
abstract type AbstractPointsIn2D <: Shape2D end

@inline plan(p::AbstractPointsIn2D) = p.plan
@inline vertices(p::AbstractPointsIn2D) = p.vertices
Base.length(p::AbstractPointsIn2D) = length(vertices(p))

# Polygon
struct ConvexPolygon <: AbstractPointsIn2D
    plan::Plan
    vertices::Vector{<:Tuple{<:Real,<:Real}}
end

#Set of points in 2D (not necessary a plogygon)
struct PointsIn2D <: AbstractPointsIn2D
    plan::Plan
    vertices::Vector{<:Tuple{<:Real,<:Real}}
end




#### Function to create shape in 3D

RectanguarCuboid(a, b, c) = ConvexPolyhedron([(a / 2, b / 2, c / 2), (-a / 2, -b / 2, -c / 2), (a / 2, -b / 2, -c / 2), (-a / 2, b / 2, -c / 2),
    (-a / 2, -b / 2, c / 2), (a / 2, b / 2, -c / 2), (-a / 2, b / 2, c / 2), (a / 2, -b / 2, c / 2)])

Cube(c) = RectanguarCuboid(c, c, c)

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

function regular_icosahedron(c) #12 vertices with triangles as faces
    # c is sidelength
    d = c / 2 #rescaling
    g = ((1 + √5) / 2) * (c / 2) # using golden ratio
    ConvexPolyhedron([(0, d, g), (0, -d, g), (0, -d, -g), (0, d, -g),
        (d, g, 0), (d, -g, 0), (-d, -g, 0), (-d, g, 0),
        (g, 0, d), (g, 0, -d), (-g, 0, -d), (-g, 0, d)])
end


