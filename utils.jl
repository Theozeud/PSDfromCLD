# Utils
_min(points::Vector, i::Int) = points[argmin([p[i] for p in points])]
_max(points::Vector,i ::Int) = points[argmax([p[i] for p in points])]
minAndmax(points::Vector, i::Int) = (_min(points, i),_max(points, i))
