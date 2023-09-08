using Distances
using LinearAlgebra

# Utils
_min(points::Vector, i::Int) = min([p[i] for p in points])
_max(points::Vector,i ::Int) = max([p[i] for p in points])
minAndmax(points::Vector, i::Int) = (_min(points, i),_max(points, i))


# Différentes formes
pave(a,b,c) = [(a/2,b/2,c/2),(-a/2,-b/2,-c/2),(a/2,-b/2,-c/2),(-a/2,b/2,-c/2),(-a/2,-b/2,c/2), (a/2,b/2,-c/2),(-a/2,b/2,c/2),(a/2,-b/2,c/2)]
cube(c) = pave(c,c,c)


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

Rotation(Φ,θ,ϕ,X) = Rx(Φ)*Ry(θ)*Rz(ϕ)*X

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

# Enveloppe convexe

function convexHull()
#euclidean((point),(point)) pour la distance euclienne
end



#_min(points,1) au lieu de :
function abs_min(listePoints) # Point d'abscisse minimale
  ptMin = listePoints[1]
  for (a,b) in listePoints
    if a < ptMin[1]
      ptMin = (a,b)
    end
  end
  return ptMin

function angle_init(pt0,pt1) #Angle initialisation
  (a,b) = pt1
  (x,y) = pt0
  return arctan(abs(a-x)/abs(b-y))

function angle(pt0,pt1,pt2) #Angle après initialisation
  (x,y) = pt0
  (a,b) = pt1
  (c,d) = pt2
  AB = euclidean(pt0,pt1)
  BC = euclidean(pt1,pt2)
  AC = euclidean(pt0,pt2)
  θ = arccos((AB^2 + BC^2 - AC^2)/(2*AB*BC)) #Formule d'Al Kashi
  return θ

function jarvis_march(listePoints)
  ptMin = abs_min(listePoints)
  dejavu = [ptMin] #liste stockant les points déjà rencontrés dans l'enveloppe
  for pt in listePoints
    pt_ang_min = #point faisant l'angle optimal avec ptMin
    ang__init_min = Inf
    if pt not in dejavu
      ang_init = angle_init(ptMin,pt)

# Choix de la corde



# Intersection

function intersect(p₁, p₂, y)
  (y-p₁1[2])*(p₂[1]-p₁[1])/(p₂[2]-p₁[2])+p₁[1]
end

function two_ext(points,i,j,method)
  # renvoie vecteur des n plus petits selon coordonnees i, trie selon coordonnee j
  p1 = _min(points,i)
  points_bis = [p for p in points if p != p1]
  p2 = _min(points_bis,i)
  if p1[j]<p2[j]
    return [p1,p2]
  else
    return [p2,p1]
  end

  
end

function two_max(points,i,j)
  # renvoie vecteur de taille n
end

function chordlength(points,h,i,j) # i=y ; j=x;
  above_h = [p for p in points if p[i]>h]
  below_h = [p for p in points if p[i]<=h]
  
  if lenght(above_h)==1
    p_above_min = _min(above_h,i)
  else
    p_above_min = nmin(above_h,2,i,j)
  end

  if lenght(below_h)==1
    p_below_max = _max(below_h,i)
  else
    p_below_max = nmax(below_h,2,i,j)
  end
  
  x_right = intersect(p_above_min[1],p_below_max[1],h)
  x_left = intersect(p_above_min[end],p_below_max[end],h)

  x_left - x_right

end


function computeCL(X::Vector, p::Plan = XY())
      Φ = 2π*rand()
      θ = π*rand()
      ϕ = π/2 * rand()
      RotX  = Rotation(Φ,θ,ϕ,X)
      ProjX = ProjectTo(p,RotX)
      ConvX = ConvexHull(ProjX)
      edgeₘᵢₙ, edgeₘₐₓ =  minAndmax(ConvX, index(p))
      yₗ = rand()*(edgeₘₐₓ - edgeₘᵢₙ) + edgeₘᵢₙ
      chord_length(p, yₗ)
end

function computeCLD(X::Vector, ntirage::Int = 1, p::Plan = XY())
  CLD = zeros(ntirage)
  for i in 1::ntirage
    push!(CLD,computeCL(p,X::Vector))
  end
  CLD
end