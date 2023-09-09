using test
include("cld.jl")

function test_chordlength(points, h)
    above_h = [p for p in points if p[2] > h]
    below_h = [p for p in points if p[2] <= h]

    p_above_min = length(above_h) == 1 ? [_min(above_h, 2)] : two_min(above_h, 2, 1)
    p_below_max = length(below_h) == 1 ? [_max(below_h, 2)] : two_max(below_h, 2, 1)

    @show x_left = intersect2D(p_above_min[1], p_below_max[1], h)
    @show x_right = intersect2D(p_above_min[end], p_below_max[end], h)
    cl = abs(x_right - x_left)


    return cl, x_left, x_right

end



