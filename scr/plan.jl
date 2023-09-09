# Projection in the plan
abstract type Plan end
struct XY <: Plan end
struct XZ <: Plan end
struct YZ <: Plan end

index(::XZ) = 1
index(::XY) = 2
index(::YZ) = 3

Xlabel(::XY) = L"x"
Xlabel(::YZ) = L"y"
Xlabel(::XZ) = L"z"
Ylabel(::XY) = L"y"
Ylabel(::YZ) = L"z"
Ylabel(::XZ) = L"z"