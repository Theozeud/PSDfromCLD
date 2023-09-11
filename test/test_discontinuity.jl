include("../scr/cld.jl")
include("../scr/plot.jl")
include("../scr/shape.jl")


cube = Cube(3)
tetrae = Tetraedron(5)

function give_abs(low, high, nbbins) #give abscisse list
    size_bin = (high - low) / nbbins
    abscisse = [0.0 for k in range(1, nbbins)]
    for index in range(1, nbbins)
        step = low + size_bin * index
        abscisse[index] = step
    end
    abscisse
end

function plothist(cld::Vector, low, high, nbbins) #plot histogram
    size_bin = (high - low) / nbbins
    bins = [0 for k in range(1, nbbins)]
    for value in cld
        bin_index = 1
        for index in range(1, nbbins)
            step = low + size_bin * index
            if step < value
                bin_index = index
            end
        end
        bins[bin_index] += 1
    end
    absc = give_abs(low, high, nbbins)
    #@show (bins[68:72], absc[68:72]) #cube
    @show (bins[58:62], absc[58:62])

    plot(absc, bins)
end



function plotShape(cp::ConvexPolyhedron, ntirage::Int=1, p::Plan=XY()) 
    #plot projection for specific chordlength
    tmp = cp
    for i in range(1, ntirage)
        Φ = 2π * rand()
        θ = π * rand()
        ϕ = π / 2 * rand()
        V = vertices(cp)
        RotV = Rotation(V, Φ, θ, ϕ)
        ProjV = projectTo(p, RotV)
        ConvV = convexHull(ProjV)
        edgeₘᵢₙ, edgeₘₐₓ = [e[index(p)] for e in minAndmax(ConvV, index(p))]
        yₗ = rand() * (edgeₘₐₓ - edgeₘᵢₙ) + edgeₘᵢₙ
        cl = chordlength(ConvV, yₗ)
        if cl > 4.14 && cl < 4.26
            @show 1
            tmp = ProjV
        end
    end
    plot2D(tmp)
end

#cld = computeCLD(shape, 10000)
#plothist(cld, 0.0, 6.0, 100)

#plotShape(cube, 10000)

cld = computeCLD(tetrae, 10000)
plothist(cld, 0.0, 6.0, 100)
