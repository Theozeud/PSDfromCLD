

# Différentes formes
cube(c) = [(c/2,c/2,c/2),(-c/2,-c/2,-c/2),(c/2,-c/2,-c/2),(-c/2,c/2,-c/2),(-c/2,-c/2,c/2),()]
pave(a,b,c) = []


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

# Projection sur le plan (X,Y)
projectToXY(X) = X[1:2]

# Enveloppe convexe

function 

# 

# Intersection



function Routine(ntirage::Int, X::Tuple)
    for i in 1::ntirage
        Φ = 2π*rand()
        θ = π*rand()
        ϕ = π/2 * rand()
        Rot_X = Rotation(Φ,θ,ϕ)
    end
    


end