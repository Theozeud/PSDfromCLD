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


x1 = [-a / 2, -b / 2,  c / 2]
x2 = [-a / 2,  b / 2,  c / 2]
x3 = [ a / 2,  b / 2,  c / 2]
x4 = [ a / 2, -b / 2,  c / 2]
x5 = [-a / 2, -b / 2, -c / 2]
x6 = [-a / 2,  b / 2, -c / 2]
x7 = [ a / 2,  b / 2, -c / 2]
x8 = [ a / 2, -b / 2, -c / 2]

PRx1 = ProjXY(Rotz * Roty * Rotx * x1)
PRx2 = ProjXY(Rotz * Roty * Rotx * x2)
PRx3 = ProjXY(Rotz * Roty * Rotx * x3)
PRx4 = ProjXY(Rotz * Roty * Rotx * x4)
PRx5 = ProjXY(Rotz * Roty * Rotx * x5)
PRx6 = ProjXY(Rotz * Roty * Rotx * x6)
PRx7 = ProjXY(Rotz * Roty * Rotx * x7)
PRx8 = ProjXY(Rotz * Roty * Rotx * x8)
PRx = [PRx1, PRx2, PRx3, PRx4, PRx5, PRx6, PRx7, PRx8]
function distance()
    D = Dict()
    for i in 1:8
        for j in (1+i):8
            formu = expand(PRx[i][2] - PRx[j][2])
            if expand(PRx[i][2] - PRx[j][2])∉keys(D)
                D[formu] = [(i,j)]
            else
                push!(D[formu],(i,j))
            end
        end
    end
    D
end


