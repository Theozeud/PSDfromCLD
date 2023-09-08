using Plots

function plotCLD(cld::Vector)
    histogram(cld)
    title!("Chord Length Distribution")
end