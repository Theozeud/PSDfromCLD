using Symbolics
using Test


@variables ϕ

Rz = [cos(ϕ) -sin(ϕ);
      sin(ϕ) cos(ϕ) ]

@variables c

x1 = [-c/2,-c/2]
x2 = [-c/2, c/2]
x3 = [ c/2, c/2]
x4 = [ c/2,-c/2]

Rx1 = expand.(Rz*x1)
Rx2 = expand.(Rz*x2)
Rx3 = expand.(Rz*x3)
Rx4 = expand.(Rz*x4)

Rx1 = Rz*x1


intersect2D(p₁, p₂, y) = (y - p₁[2]) * (p₂[1] - p₁[1]) / (p₂[2] - p₁[2]) + p₁[1]

function yl(p1,p2,p3,p4,l)
    A₁₂ = (p2[1]-p1[1])/(p2[2]-p1[2]) 
    A₃₄ = (p4[1]-p3[1])/(p4[2]-p3[2])
    expand((l - (p1[1]-p3[1]) + A₁₂*p1[2] - A₃₄*p3[2])/(A₁₂ - A₃₄))
end

@variables l
y = yl(Rx2,Rx1,Rx4,Rx1,l)

Dl = Differential(l)

expand_derivatives(Dl(y))