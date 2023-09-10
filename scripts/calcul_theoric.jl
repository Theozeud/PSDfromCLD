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
# Dans ce cas, Conv(Φ, θ) = {x₁,x₂,x₃,x₅,x₇,x₈}
# Dans cette configuration, on a en terme de coordonnées sur l'axe y : 
#                        x₂≥x₃≥x₇≥x₁≥x₅≥x₈.

# Les hauteurs ymax et ymin sont:
ymax = PRx₂[2]
ymin = PRx₈[2]
Ptot = ymax - ymin

# Fonction utile qui renvoie l'abscisse d'un point dont on connait l'ordonnée sur la droite formée par les points p₁ et p₂
intersect2D(p₁, p₂, y) = (y - p₁[2]) * (p₂[1] - p₁[1]) / (p₂[2] - p₁[2]) + p₁[1]

function yₗ(p1,p2,p3,p4,l)
    A₁₂ = (p2[1]-p1[1])/(p2[2]-p1[2]) 
    A₃₄ = (p4[1]-p3[1])/(p4[2]-p3[2])
    (l - (p1[1]-p3[1]) + A₁₂*p1[2] - A₃₄*p3[2])/(A₁₂ - A₃₄)
end

@variables y,l

# Calcul de la longueur de corde entre x₂ et x₃ et de la probabilité d'être dans cette intervalle 
# Elle est constante sur cette intervalle.
l₃₂ = expand(intersect2D(PRx₂,PRx₃,y) - intersect2D(PRx₃,PRx₇,y))
y₃₂ = yₗ(PRx₂,PRx₃,PRx₇,PRx₃,l)
P₃₂ = (PRx₃[2]-y)/(Ptot)

# Calcul de la longueur de corde entre x₂ et x₇ et de la probabilité d'être dans cette intervalle 
l₂₇ = expand(intersect2D(PRx₇,PRx₃,y)-intersect2D(PRx₂,PRx₁,y))
y₂₇ = yₗ(PRx₇,PRx₃,PRx₂,PRx₁,l)
P₂₇ = (PRx₂[2]-y)/(Ptot)

# Calcul de la longueur de corde entre x₇ et x₁ et de la probabilité d'être dans cette intervalle 
l₇₁ = expand(intersect2D(PRx₅,PRx₇,y) - intersect2D(PRx₂,PRx₁,y))
y₇₁ = yₗ(PRx₅,PRx₇,PRx₂,PRx₁,l)
P₇₁ = (PRx₇[2]-y)/(Ptot)


# Calcul de la probabilité
fun_l₃₂ = eval(build_function(l₃₂,y,c,Φ,θ,ϕ))
fun_l₂₇ = eval(build_function(l₂₇,y,c,Φ,θ,ϕ))
fun_l₇₁ = eval(build_function(l₇₁,y,c,Φ,θ,ϕ))
fun_P₃₂ = eval(build_function(P₃₂,y,c,Φ,θ,ϕ))
fun_P₂₇ = eval(build_function(P₂₇,y,c,Φ,θ,ϕ))
fun_P₇₁ = eval(build_function(P₇₁,y,c,Φ,θ,ϕ))
fun_y₃₂ = eval(build_function(y₃₂,l,c,Φ,θ,ϕ))
fun_y₂₇ = eval(build_function(y₂₇,l,c,Φ,θ,ϕ))
fun_y₇₁ = eval(build_function(y₇₁,l,c,Φ,θ,ϕ))

fun_x₁ = eval(build_function(PRx₁,c,Φ,θ,ϕ)[1])
fun_x₂ = eval(build_function(PRx₂,c,Φ,θ,ϕ)[1])
fun_x₃ = eval(build_function(PRx₃,c,Φ,θ,ϕ)[1])
fun_x₄ = eval(build_function(PRx₄,c,Φ,θ,ϕ)[1])
fun_x₅ = eval(build_function(PRx₅,c,Φ,θ,ϕ)[1])
fun_x₆ = eval(build_function(PRx₆,c,Φ,θ,ϕ)[1])
fun_x₇ = eval(build_function(PRx₇,c,Φ,θ,ϕ)[1])
fun_x₈ = eval(build_function(PRx₈,c,Φ,θ,ϕ)[1])

function Proba(l,c,Φ,θ,ϕ)
    x₇ = fun_x₃(c,Φ,θ,ϕ)
    x₂   = fun_x₂(c,Φ,θ,ϕ)
    if l ≤ fun_l₃₂(x₂[2],c,Φ,θ,ϕ)
        p = fun_P₃₂(fun_y₃₂(l,c,Φ,θ,ϕ),c,Φ,θ,ϕ)
    elseif l≤fun_l₂₇(x₇[2],c,Φ,θ,ϕ)
        p = fun_P₃₂(x₂[2],c,Φ,θ,ϕ)+fun_P₂₇(fun_y₂₇(l,c,Φ,θ,ϕ),c,Φ,θ,ϕ)
    else 
        p = fun_P₃₂(x₂[2],c,Φ,θ,ϕ)+fun_P₆₃(x₇[2],c,Φ,θ,ϕ)+fun_P₃₂(fun_y₇₁(l,c,Φ,θ,ϕ),c,Φ,θ,ϕ)
    end
    p
end

# Tests

Φ = 0.2
θ = 0.2
ϕ = 0.1
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

# Pour Φ,θ,ϕ ≤ π/4
@test x₂[2]≥x₃[2]≥x₇[2]≥x₁[2]≥x₅[2]≥x₈[2]

fun_l₃₂(x₂[2],c,Φ,θ,ϕ)
Proba(1,2,0.1,0.1,0.1)