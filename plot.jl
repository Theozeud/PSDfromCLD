using Plots

function plotCLD(cld::Vector)
    histogram(cld,bins = range(0,6,100))
    title!("Chord Length Distribution")
end