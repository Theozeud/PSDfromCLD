using Symbolics
using Test

include("../scr/rotation.jl")

@variables Φ,θ,ϕ

Rotx = Rx(Φ) 
Roty = Ry(θ)
Rotz = Rz(ϕ)
ProjXY(X) = X[1:2]

@variables c
a = c
b = c


x₁ = [-a / 2, -b / 2,  c / 2]
x₂ = [-a / 2,  b / 2,  c / 2]
x₃ = [ a / 2,  b / 2,  c / 2]
x₄ = [ a / 2, -b / 2,  c / 2]
x₅ = [-a / 2, -b / 2, -c / 2]
x₆ = [-a / 2,  b / 2, -c / 2]
x₇ = [ a / 2,  b / 2, -c / 2]
x₈ = [ a / 2, -b / 2, -c / 2]

PRx₁ = ProjXY(Rotz * Roty * Rotx * x₁)
PRx₂ = ProjXY(Rotz * Roty * Rotx * x₂)
PRx₃ = ProjXY(Rotz * Roty * Rotx * x₃)
PRx₄ = ProjXY(Rotz * Roty * Rotx * x₄)
PRx₅ = ProjXY(Rotz * Roty * Rotx * x₅)
PRx₆ = ProjXY(Rotz * Roty * Rotx * x₆)
PRx₇ = ProjXY(Rotz * Roty * Rotx * x₇)
PRx₈ = ProjXY(Rotz * Roty * Rotx * x₈)

# On suppose 0≤ Φ,θ,ϕ < π/4
# Dans ce cas, Conv(Φ, θ) = {x₁,x₃,x₄,x₅,x₆,x₇}
# Dans cette configuration, on a en terme de coordonnées sur l'axe y : 
#                        x₇≥x₃≥x₆≥x₄≥x₅≥x₁.


# Les hauteurs ymax et ymin sont:
ymax = PRx₇[2]
ymin = PRx₁[2]
Ptot = expand(ymax - ymin)

# Fonction utile qui renvoie l'abscisse d'un point dont on connait l'ordonnée sur la droite formée par les points p₁ et p₂
intersect2D(p₁, p₂, y) = (y - p₁[2]) * (p₂[1] - p₁[1]) / (p₂[2] - p₁[2]) + p₁[1]

function yₗ(p1,p2,p3,p4,l)
    A₁₂ = (p2[1]-p1[1])/(p2[2]-p1[2]) 
    A₃₄ = (p4[1]-p3[1])/(p4[2]-p3[2])
    (l - (p1[1]-p3[1]) + A₁₂*p1[2] - A₃₄*p3[2])/(A₁₂ - A₃₄)
end

@variables y,l

# Calcul de la longueur de corde entre x₇ et x₃ et de la probabilité d'être dans cette intervalle 
# Elle est constante sur cette intervalle.
l₇₃ = expand(intersect2D(PRx₇,PRx₃,y) - intersect2D(PRx₆,PRx₇,y))
y₇₃ = yₗ(PRx₃,PRx₇,PRx₆,PRx₇,l)
P₇₃ = expand((PRx₇[2]-y)/(Ptot))

# Calcul de la longueur de corde entre x₃ et x₆ et de la probabilité d'être dans cette intervalle 
l₃₆ = expand(intersect2D(PRx₃,PRx₄,y)-intersect2D(PRx₇,PRx₆,y))
y₃₆ = yₗ(PRx₃,PRx₄,PRx₇,PRx₆,l)
P₃₆ = expand((PRx₃[2]-y)/(Ptot))

# Calcul de la longueur de corde entre x₆ et x₄ et de la probabilité d'être dans cette intervalle 
l₆₄ = expand(intersect2D(PRx₃,PRx₄,y) - intersect2D(PRx₆,PRx₅,y))
y₆₄ = yₗ(PRx₃,PRx₄,PRx₆,PRx₅,l)
P₆₄ = expand((PRx₆[2]-y)/(Ptot))


# Calcul de la probabilité
fun_l₇₃ = eval(build_function(l₇₃,y,c,Φ,θ,ϕ))
fun_l₃₆ = eval(build_function(l₃₆,y,c,Φ,θ,ϕ))
fun_l₆₄ = eval(build_function(l₆₄,y,c,Φ,θ,ϕ))
fun_P₇₃ = eval(build_function(P₇₃,y,c,Φ,θ,ϕ))
fun_P₃₆ = eval(build_function(P₃₆,y,c,Φ,θ,ϕ))
fun_P₆₄ = eval(build_function(P₆₄,y,c,Φ,θ,ϕ))
fun_y₇₃ = eval(build_function(y₇₃,l,c,Φ,θ,ϕ))
fun_y₃₆ = eval(build_function(y₃₆,l,c,Φ,θ,ϕ))
fun_y₆₄ = eval(build_function(y₆₄,l,c,Φ,θ,ϕ))

fun_x₁ = eval(build_function(PRx₁,c,Φ,θ,ϕ)[1])
fun_x₂ = eval(build_function(PRx₂,c,Φ,θ,ϕ)[1])
fun_x₃ = eval(build_function(PRx₃,c,Φ,θ,ϕ)[1])
fun_x₄ = eval(build_function(PRx₄,c,Φ,θ,ϕ)[1])
fun_x₅ = eval(build_function(PRx₅,c,Φ,θ,ϕ)[1])
fun_x₆ = eval(build_function(PRx₆,c,Φ,θ,ϕ)[1])
fun_x₇ = eval(build_function(PRx₇,c,Φ,θ,ϕ)[1])
fun_x₈ = eval(build_function(PRx₈,c,Φ,θ,ϕ)[1])

function Proba(l,c,Φ,θ,ϕ)
    x₆ = fun_x₆(c,Φ,θ,ϕ)
    x₃ = fun_x₃(c,Φ,θ,ϕ)
    p = 0
    if l ≤ fun_l₇₃(x₃[2],c,Φ,θ,ϕ)
        p = fun_P₇₃(fun_y₇₃(l,c,Φ,θ,ϕ),c,Φ,θ,ϕ)
    elseif l≤fun_l₃₆(x₆[2],c,Φ,θ,ϕ)
        p = fun_P₇₃(x₃[2],c,Φ,θ,ϕ)+fun_P₃₆(fun_y₃₆(l,c,Φ,θ,ϕ),c,Φ,θ,ϕ)
    else 
        p = fun_P₇₃(x₃[2],c,Φ,θ,ϕ)+fun_P₃₆(x₆[2],c,Φ,θ,ϕ)+fun_P₆₄(max(fun_y₆₄(l,c,Φ,θ,ϕ),0),c,Φ,θ,ϕ)
    end
    2*p
end

using Test

Φ = 0.2
θ = 0.2
ϕ = 0.79
c = 1
l = 1



x₁ = fun_x₁(c,Φ,θ,ϕ)
x₂ = fun_x₂(c,Φ,θ,ϕ)
x₃ = fun_x₃(c,Φ,θ,ϕ)
x₄ = fun_x₄(c,Φ,θ,ϕ)
x₅ = fun_x₅(c,Φ,θ,ϕ)
x₆ = fun_x₆(c,Φ,θ,ϕ)
x₇ = fun_x₇(c,Φ,θ,ϕ)
x₈ = fun_x₈(c,Φ,θ,ϕ)






#=
# Pour Φ,θ,ϕ ≤ π/4
@test x₇[2]≥x₃[2]≥x₆[2]≥x₄[2]≥x₅[2]≥x₁[2]


# Tests pour vérifier que l(y(l)) = L et y(l(y)) = yi
@test abs(fun_y₇₃(fun_l₇₃(x₃[2],c,Φ,θ,ϕ),c,Φ,θ,ϕ)-fun_l₇₃(fun_y₇₃(x₃[2],c,Φ,θ,ϕ),c,Φ,θ,ϕ)) < 1e-13
@test abs(fun_y₇₃(fun_l₇₃(x₃[2],c,Φ,θ,ϕ),c,Φ,θ,ϕ)-x₃[2]) < 1e-13

# Tests pour croissance de la corde
@test fun_l₇₃(x₃[2],c,Φ,θ,ϕ) ≤ fun_l₃₆(x₆[2],c,Φ,θ,ϕ) ≤ fun_l₆₄(x₄[2],c,Φ,θ,ϕ)



Proba(1,c,Φ,θ,ϕ)

lrange = 0:0.01:1.5

using Plots
include("../scr/utils.jl")


include("../scr/cld.jl")
include("../scr/shape.jl")
include("../scr/plot.jl")

shape = Cube(c)
cld = computeCLD(shape, 1000; Φ = Φ, θ = θ, ϕ = ϕ)
plt2 = plotCumulCLD(cld/1000)

plot!(plt2, lrange, apply(l->Proba(l,c,Φ,θ,ϕ),lrange))


rangeΦ = 0:0.01:2π
rangeθ = 0:0.005:π

function plot_convexhull(c,ϕ)
    X = zeros(length(rangeΦ),length(rangeθ))
    i = 0
    j = 0
    for Φ in rangeΦ
        i+=1
        for θ in rangeθ
            j+=1
            x₁ = fun_x₁(c,Φ,θ,ϕ)
            x₂ = fun_x₂(c,Φ,θ,ϕ)
            x₃ = fun_x₃(c,Φ,θ,ϕ)
            x₄ = fun_x₄(c,Φ,θ,ϕ)
            x₅ = fun_x₅(c,Φ,θ,ϕ)
            x₆ = fun_x₆(c,Φ,θ,ϕ)
            x₇ = fun_x₇(c,Φ,θ,ϕ)
            x₈ = fun_x₈(c,Φ,θ,ϕ)
            cv = convexHull([x₁,x₂,x₃,x₄,x₅,x₆,x₇,x₈])
            if x₁∉cv
                X[i,j] = 1
            elseif x₂∉cv
                X[i,j] = 2
            elseif x₃∉cv
                X[i,j] = 3
            elseif x₄∉cv
                X[i,j] = 4
            else
                X[i,j] = 0
            end     
        end  
        j = 0
        end
    X
end

P = plot_convexhull(3,0)

using Plots
Plots.plot3d(rangeΦ,rangeθ,P)
=#