using Distances
using LinearAlgebra

include("shape.jl")
include("utils.jl")


# Utils for ConvexHull
  
function angle_init(pt0,pt1) #Angle initialisation
    (x,y) = pt0
    (a,b) = pt1
    atan(abs(a-x)/abs(b-y))
end

function _angle(pt0, pt1, pt2) #Angle après initialisation
    AB = euclidean(pt0, pt1)
    BC = euclidean(pt1, pt2)
    AC = euclidean(pt0, pt2)
    #Formule d'Al Kashi
    L = (AB^2 + BC^2 - AC^2) / (2 * AB * BC)
    # Correction in case of micro error
    L = norm(L) > 1 ? L / norm(L) : L
    acos(L)
end



# Method to compute the convex hull of a set of points using the algorithm of Javis' march.
convexHull(p::ConvexPolygon) = p
convexHull(v::PointsIn2D) = Polygon(plan(v), convexHull(vertices(v)))

function convexHull(Points::Vector)
    ptMin = _min(Points, 1)
    conv = eltype(Points)[] # points we have already met

    pt_ang_min = (0, 0) # points with optimal angle with ptMin
    ang_init_min = Inf

    for pt in Points #second point for initialisation
        if pt ∉ conv
            ang_init = angle_init(ptMin, pt)
            if ang_init < ang_init_min
                ang_init_min = ang_init
                pt_ang_min = pt
            end #then we got the point with optimal angle
        end
    end

    beforePt = ptMin #point before current point
    currentPt = pt_ang_min # current point in convex hull

    while currentPt != ptMin
        optim_ang = -Inf #optimum angle with the three current points
        optim_pt = (0, 0)
        for pt in Points
            if pt ∉ conv && pt != currentPt
                test_ang = _angle(beforePt, currentPt, pt)  #angle to test
                if test_ang > optim_ang
                    optim_ang = test_ang
                    optim_pt = pt
                end
            end
        end
        beforePt = currentPt
        currentPt = optim_pt
        push!(conv, beforePt)
    end

    push!(conv, ptMin)
    conv
end