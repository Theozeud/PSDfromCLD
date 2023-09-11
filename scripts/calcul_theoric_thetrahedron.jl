using Symbolics
using Test

include("../scr/rotation.jl")

@variables Φ,θ,ϕ

Rotx = Rx(Φ) 
Roty = Ry(θ)
Rotz = Rz(ϕ)
ProjXY(X) = X[1:2]

@variables c

co = c / (2 * √2)

x₁ = [ co,  co,  co]
x₂ = [ co, -co, -co]
x₃ = [-co,  co, -co]
x₄ = [-co, -co,  co]


PRx₁ = ProjXY(Rotz * Roty * Rotx * x₁)
PRx₂ = ProjXY(Rotz * Roty * Rotx * x₂)
PRx₃ = ProjXY(Rotz * Roty * Rotx * x₃)
PRx₄ = ProjXY(Rotz * Roty * Rotx * x₄)


# Fonction utile qui renvoie l'abscisse d'un point dont on connait l'ordonnée sur la droite formée par les points p₁ et p₂
intersect2D(p₁, p₂, y) = (y - p₁[2]) * (p₂[1] - p₁[1]) / (p₂[2] - p₁[2]) + p₁[1]

function yₗ(p1,p2,p3,p4,l)
    A₁₂ = (p2[1]-p1[1])/(p2[2]-p1[2]) 
    A₃₄ = (p4[1]-p3[1])/(p4[2]-p3[2])
    (l - (p1[1]-p3[1]) + A₁₂*p1[2] - A₃₄*p3[2])/(A₁₂ - A₃₄)
end

@variables y,l


# Calcul de la probabilité


fun_x₁ = eval(build_function(PRx₁,c,Φ,θ,ϕ)[1])
fun_x₂ = eval(build_function(PRx₂,c,Φ,θ,ϕ)[1])
fun_x₃ = eval(build_function(PRx₃,c,Φ,θ,ϕ)[1])
fun_x₄ = eval(build_function(PRx₄,c,Φ,θ,ϕ)[1])


include("../scr/convexhull.jl")

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
            cv = convexHull([x₁,x₂,x₃,x₄])
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