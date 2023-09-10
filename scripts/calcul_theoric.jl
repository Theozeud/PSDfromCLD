using Symbolics

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

# On suppose Φ, θ ∈ [0,-π/2[
# Dans ce cas, Conv(Φ, θ) = {x₁,x₃,x₄,x₅,x₆,x₇}
# Dans cette configuration, on a en terme de coordonnées sur l'axe y : 
#                        x₇≥x₆≥x₃≥x₅≥x₄≥x₁.

# Les hauteurs ymax et ymin sont:
ymax = PRx₇[2]
ymin = PRx₁[2]
Ptot = ymax - ymin

# Fonction utile qui renvoie l'abscisse d'un point dont on connait l'ordonnée sur la droite formée par les points p₁ et p₂
intersect2D(p₁, p₂, y) = (y - p₁[2]) * (p₂[1] - p₁[1]) / (p₂[2] - p₁[2]) + p₁[1]

@variables y

# Calcul de la longueur de corde entre x₃ et x₅ et de la probabilité d'être dans cette intervalle 
# Elle est constante sur cette intervalle.
l₃₅ = expand(intersect2D(PRx₃,PRx₄,y) - intersect2D(PRx₆,PRx₅,y))
P₃₅ = (PRx₃[2]-PRx₅[2])/(Ptot)

# Calcul de la longueur de corde entre x₆ et x₃ et de la probabilité d'être dans cette intervalle 
l₆₃ = expand(intersect2D(PRx₇,PRx₃,y)-intersect2D(PRx₆,PRx₅,y))
P₆₃ = PRx₆[2]-PRx₃[2]/(Ptot)

# Calcul de la longueur de corde entre x₇ et x₆ et de la probabilité d'être dans cette intervalle 
l₇₆ = expand(intersect2D(PRx₇,PRx₃,y) - intersect2D(PRx₇,PRx₆,y))
P₇₆ = (PRx₇[2]-PRx₆[2])/(Ptot)


# Calcul de la probabilité
fun₃₅ = eval(build_function(l₃₅,y))
fun₆₃ = eval(build_function(l₆₃,y))
fun₇₆ = eval(build_function(l₇₆,y))

function Proba(l,L)



end