using Plots

function plotCLD(cld::Vector)
    histogram(cld,bins = range(0,10,100))
    title!("Chord Length Distribution")
end