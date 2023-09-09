# Projection in the plan

abstract type Plan end
struct XY <: Plan end
struct XZ <: Plan end
struct YZ <: Plan end

projectTo(::XY,X) = X[1:2]
projectTo(::XZ,X) = X[1:2:3]
projectTo(::YZ,X) = X[2:3]

index(::XY) = 2
index(::XZ) = 1
index(::YZ) = 3