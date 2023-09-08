using Distances
using LinearAlgebra
using Test

# Utils
_min(points::Vector, i::Int) = points[argmin([p[i] for p in points])]
_max(points::Vector,i ::Int) = points[argmax([p[i] for p in points])]
minAndmax(points::Vector, i::Int) = (_min(points, i),_max(points, i))


# Différentes formes
pave(a,b,c) = [(a/2,b/2,c/2),(-a/2,-b/2,-c/2),(a/2,-b/2,-c/2),(-a/2,b/2,-c/2),(-a/2,-b/2,c/2), (a/2,b/2,-c/2),(-a/2,b/2,c/2),(a/2,-b/2,c/2)]
cube(c) = pave(c,c,c)

@test length(pave(1,1,1)) == 8

# Rotation
Rx(Φ) = [1   0      0;
         0  cos(Φ) sin(Φ);
         0 -sin(Φ) cos(Φ)]

Ry(θ) = [cos(θ)  0 sin(θ);
         0       1   0;
         -sin(θ) 0 cos(θ)]
       
Rz(ϕ) = [cos(ϕ) sin(ϕ) 0;
         -sin(ϕ) cos(ϕ) 0;
           0     0     1]

Rotation(Φ,θ,ϕ,X) = Rx(Φ)*Ry(θ)*Rz(ϕ)*[X...]

# Projection sur le plan

abstract type Plan end
struct XY <: Plan end
struct XZ <: Plan end
struct YZ <: Plan end

projectTo(::XY,X) = X[1:2]
projectTo(::XZ,X) = X[1:2:3]
projectTo(::YZ,X) = X[2:3]

index(::XY) = 2
index(::XZ) = 1
index(::YZ) = 3

# ConvexHull

function convexHull(points)
  jarvis_march(points)
end


function angle_init(pt0,pt1) #Angle initialisation
  (x,y) = pt0
  (a,b) = pt1
  atan(abs(a-x)/abs(b-y))
end

function _angle(pt0,pt1,pt2) #Angle après initialisation
  AB = euclidean(pt0,pt1)
  BC = euclidean(pt1,pt2)
  AC = euclidean(pt0,pt2)
  acos((AB^2 + BC^2 - AC^2)/(2*AB*BC)) #Formule d'Al Kashi
end

function jarvis_march(listePoints)
  ptMin = _min(listePoints,1)
  conv = [] #points we have already met
  
  pt_ang_min = (0,0)#points with optimal angle with ptMin
  ang_init_min = Inf

  for pt in listePoints #second point for initialisation
    if pt ∉ conv
      ang_init = angle_init(ptMin,pt)
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
    optim_pt = (0,0)
    for pt in listePoints
      if pt ∉ conv
        if pt != currentPt
          test_ang = _angle(beforePt, currentPt, pt)  #angle to test
          if test_ang > optim_ang
            optim_ang = test_ang
            optim_pt = pt
          end
        end
      end
    end
    beforePt = currentPt
    currentPt = optim_pt
    push!(conv,beforePt)
  end
  push!(conv, ptMin)
  return conv
end

# Chord choice



# Intersection
function intersect2D(p₁, p₂, y)
  @assert length(p₁) == length(p₂) == 2
  (y-p₁[2])*(p₂[1]-p₁[1])/(p₂[2]-p₁[2])+p₁[1]
end

function two_min(points,i,j)
  # renvoie vecteur des n plus petits selon coordonnees i, trie selon coordonnee j
  p1 = _min(points,i)
  points_bis = [p for p in points if p != p1]
  p2 = _min(points_bis,i)
  p1[j]<p2[j] ? [p1,p2] : [p2,p1]
end

function two_max(points,i,j)
  p1 = _max(points,i)
  points_bis = [p for p in points if p != p1]
  p2 = _max(points_bis,i)
  p1[j]<p2[j] ? [p1,p2] : [p2,p1]
end

function chordlength(points,h) # i=y ; j=x;
  above_h = [p for p in points if p[2]>h]
  below_h = [p for p in points if p[2]<=h]
  
  p_above_min = length(above_h)==1 ? [_min(above_h,2)] : two_min(above_h,2,1)
  p_below_max = length(below_h) ==1 ? [_max(below_h,2)] : two_max(below_h,2,1)
  
  x_left = intersect2D(p_above_min[1],p_below_max[1],h)
  x_right = intersect2D(p_above_min[end],p_below_max[end],h)

  abs(x_right - x_left)

end

carre = [(-1,1),(-1,-1),(1,1),(1,-1)]
@test chordlength(carre,0) == 2

triangle = [(-1, 0), (1, 0), (0, 1)]
@test chordlength(triangle, 0) == 2


function computeCL(X::Vector, p::Plan = XY())
      Φ = 0 # 2π*rand()
      θ = 0 # π*rand()
      ϕ = 0 # π/2 * rand()
      RotX  = [Rotation(Φ,θ,ϕ,x) for x in X]
      ProjX = [projectTo(p,x) for x in RotX]
      ConvX = convexHull(ProjX)
      edgeₘᵢₙ, edgeₘₐₓ =  [e[index(p)] for e in minAndmax(ConvX, index(p))]
      yₗ = rand()*(edgeₘₐₓ - edgeₘᵢₙ) + edgeₘᵢₙ
      chordlength(ConvX, yₗ)
end

function computeCLD(X::Vector, ntirage::Int = 1, p::Plan = XY())
  CLD = zeros(ntirage)
  for i in 1:ntirage
    CLD[i] = computeCL(X,p)
  end
  CLD
end



cubes = cube(4)
cld = computeCLD(cubes,1000)

include("plot.jl")

plotCLD(cld)

