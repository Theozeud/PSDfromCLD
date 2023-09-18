include("../scr/shape.jl")
include("../scr/cld.jl")
include("../scr/plot.jl")



function plotPeaks(nbtirage, nbbins)
    c_tetra = rand() * 3 + 2
    c_cube = rand() * 1 + 2
    cube = Cube(c_cube)
    tetra = Tetraedron(c_tetra)
    cld_cube = computeCLD(cube, nbtirage)
    cld_tetra = computeCLD(tetra, nbtirage)
    cld_union = cld_cube ∪ cld_tetra
    low = 0
    high = 6
    absc = range(low, high, nbbins)
    bins = zeros(nbbins)
    for value in cld_union
        bin_index = 1
        for (index, step) in zip(1:nbbins, absc)
            if step < value
                bin_index = index
            end
        end
        bins[bin_index] += 1
    end
    @show c_cube, c_tetra
    plot(absc, bins)
    vline!([(√(2) / 2) * c_tetra, c_cube, √(2) * c_cube])
end


function plotPeaks_onlycube(nbtirage, nbbins)
    c_tetra = 3
    c_cube = 3
    cube = Cube(c_cube)
    tetra = Tetraedron(c_tetra)
    cld_cube = computeCLD(cube, nbtirage)
    cld_tetra = computeCLD(tetra, nbtirage)
    cld_union = cld_cube
    low = 0
    high = 6
    absc = range(low, high, nbbins)
    bins = zeros(nbbins)
    for value in cld_union
        bin_index = 1
        for (index, step) in zip(1:nbbins, absc)
            if step < value
                bin_index = index
            end
        end
        bins[bin_index] += nbbins / ((high - low) * nbtirage)
    end
    @show c_cube, c_tetra
    plot(absc, bins, linewidth=3)
    title!("CLD derivative of a cube of edge length 3")
    xlabel!("Chord length")
    ylabel!("P(l = L)")
    #vline!([(√(2) / 2) * c_tetra, c_cube, √(2) * c_cube])
    #vline!([c_cube, √(2) * c_cube])
end

function plotPeaks_onlytetra(nbtirage, nbbins)
    c_tetra = 5
    c_cube = 3
    cube = Cube(c_cube)
    tetra = Tetraedron(c_tetra)
    cld_cube = computeCLD(cube, nbtirage)
    cld_tetra = computeCLD(tetra, nbtirage)
    cld_union = cld_tetra
    low = 0
    high = 6
    absc = range(low, high, nbbins)
    bins = zeros(nbbins)
    for value in cld_union
        bin_index = 1
        for (index, step) in zip(1:nbbins, absc)
            if step < value
                bin_index = index
            end
        end
        bins[bin_index] += nbbins / ((high - low) * nbtirage)
    end
    @show c_cube, c_tetra
    plot(absc, bins, linewidth=3)
    title!("CLD derivative of a tetrahedron of edge length 5")
    xlabel!("Chord length")
    ylabel!("P(l = L)")
    #vline!([(√(2) / 2) * c_tetra, c_cube, √(2) * c_cube])
    #vline!([c_cube, √(2) * c_cube])
end


function find_sizes(nbtirage, nbbins)
    c_tetra = rand() * 3 + 2
    c_cube = rand() * 1 + 2
    @show (c_tetra, c_cube)
    @show ((√(2) / 2) * c_tetra, c_cube, (√2) * c_cube)
    cube = Cube(c_cube)
    tetra = Tetraedron(c_tetra)
    cld_cube = computeCLD(cube, nbtirage)
    cld_tetra = computeCLD(tetra, nbtirage)
    cld_union = cld_cube ∪ cld_tetra
    low = 0
    high = 6
    absc = range(low, high, nbbins)
    bins = zeros(nbbins)
    for value in cld_union
        bin_index = 1
        for (index, step) in zip(1:nbbins, absc)
            if step < value
                bin_index = index
            end
        end
        bins[bin_index] += 1
    end
    #plot(absc, bins)
    dsec = zeros(nbbins)#second derivative
    for i in range(2, nbbins - 1)
        dsec[i] = -(bins[i+1] - 2 * bins[i] + bins[i-1])
    end
    plt = plot(absc, dsec)
    sorted_dlength = sortperm(dsec, rev=true) * ((high - low) / nbbins)
    disc = sorted_dlength[1:3]  #discontinuity points
    minvalue = Inf
    iinit = 1
    jinit = 2
    for i in range(1, 3)
        for j in range(1, 3)
            if i != j
                newvalue = abs(disc[i] - (√2) * disc[j])
                if newvalue < minvalue
                    minvalue = newvalue
                    iinit = i
                    jinit = j
                end
            end
        end
    end
    @show (disc[iinit], disc[jinit])
    #so iinit corresponds to the higher cube peak and jinit the lower
    cube_edge_length = disc[jinit]
    tetra_edge_length = disc[[setdiff(Set([1, 2, 3]), Set([iinit, jinit]))...][1]] * (2 / √2)
    @show (√2 / 2) * c_tetra
    @show (c_cube, c_tetra, cube_edge_length, tetra_edge_length)
    return plt
end


plotPeaks_onlycube(200000, 200)

#find_sizes(100000, 200)