## Phenomenological-RG crossing plot.
##
## The effective exponent between two sizes,
##      eta(L1,L2; Delta) = log[ Q(L1)/Q(L2) ] / log(L2/L1),
## is the local slope of log Q against log L. At a critical point the decay is
## scale invariant, so eta becomes L-independent and the curves for different
## (L1,L2) pairs cross there. Unlike the L^eta Q plots this needs NO exponent as
## input, so the crossing location is an output rather than an assumption.
##
## Caveat that drives the pair choice: the separation r = N/2 alternates parity
## along N = 6,8,10,12, and eta oscillates strongly with it. Only same-parity
## pairs are meaningful - hence (6,10) and (8,12).
##
##     julia xxz_plot_crossing.jl [name] [folder]

using Plots, LaTeXStrings, Printf
include(joinpath(@__DIR__, "xxz_plot_renyi2.jl"))

function eta_curves(A, N, D, pairs)
    idx = Dict(n => i for (i, n) in enumerate(N))
    [(p, [log(A[idx[p[1]], j] / A[idx[p[2]], j]) / log(p[2] / p[1]) for j in eachindex(D)])
     for p in pairs]
end

function find_crossing(D, y1, y2)
    ## linear interpolation of the first sign change of y1 - y2
    d = y1 .- y2
    for k in 1:length(d)-1
        if d[k] * d[k+1] < 0
            t = d[k] / (d[k] - d[k+1])
            return D[k] + t*(D[k+1] - D[k])
        end
    end
    return NaN
end

function plot_crossing(name="fidelity_p0.2";
                       same=[(6,10), (8,12)], mixed=[(6,8), (8,10), (10,12)],
                       folder="save_results/xxz", outdir="figures",
                       ysym=raw"F(\rho,\, X_iX_j\rho X_jX_i)", tag="crossing")
    A, N, D, p = read_array_json(joinpath(folder, name * ".json"))
    mkpath(outdir)
    cs = eta_curves(A, N, D, same)
    cm = eta_curves(A, N, D, mixed)
    Dc = find_crossing(D, cs[1][2], cs[2][2])
    eta_c = isnan(Dc) ? NaN :
        let k = findfirst(j -> D[j] >= Dc, eachindex(D))
            k === nothing || k == 1 ? cs[1][2][end] :
            let t = (Dc - D[k-1])/(D[k] - D[k-1]); cs[1][2][k-1] + t*(cs[1][2][k] - cs[1][2][k-1]) end
        end
    @printf("%s: same-parity crossing at Delta = %.3f, eta = %.3f\n", name, Dc, eta_c)

    ## (a) the exponent curves themselves
    p1 = plot(xlabel = L"\Delta", ylabel = L"\eta_{\mathrm{eff}} = -\,\mathrm{d}\log Q/\mathrm{d}\log L",
              title = "effective exponent, no rescaling assumed", legend = :topright,
              framestyle = :box, size = (640, 480), titlefontsize = 10,
              guidefontsize = 11, legendfontsize = 8)
    for (pr, y) in cm
        plot!(p1, D, y, lc = :gray, ls = :dot, lw = 1.2, label = L"(%$(pr[1]),%$(pr[2]))\ \mathrm{mixed}\ r")
    end
    for ((pr, y), c) in zip(cs, (:royalblue, :crimson))
        plot!(p1, D, y, lc = c, lw = 2, marker = :circle, markersize = 4,
              label = L"(%$(pr[1]),%$(pr[2]))\ \mathrm{same}\ r")
    end
    if !isnan(Dc)
        scatter!(p1, [Dc], [eta_c], mc = :black, ms = 7, marker = :star5,
                 label = L"\Delta_\times = %$(round(Dc, digits=2))")
        vline!(p1, [Dc], lc = :black, ls = :dash, alpha = 0.5, label = "")
    end

    ## (b) the same statement as a rescaled plot, using eta AT the crossing
    p2 = plot(xlabel = L"\Delta", ylabel = latexstring("L^{$(round(eta_c,digits=2))}\\, $(ysym)"),
              title = "rescaled with the crossing exponent", legend = :topleft,
              framestyle = :box, size = (640, 480), titlefontsize = 10,
              guidefontsize = 11, legendfontsize = 8)
    marks = [:circle, :square, :utriangle, :diamond]
    for (i, n) in enumerate(N)
        ## dashed = odd r, solid = even r, so the parity families are visible
        plot!(p2, D, A[i, :] .* n^eta_c, label = L"L = %$(n)",
              marker = marks[mod1(i, 4)], markersize = 4, lw = 1.8,
              ls = isodd(n ÷ 2) ? :dash : :solid)
    end
    !isnan(Dc) && vline!(p2, [Dc], lc = :black, ls = :dash, alpha = 0.5, label = "")

    pl = plot(p1, p2, layout = (1, 2), size = (1180, 470))
    for ext in ("png", "pdf")
        savefig(pl, joinpath(outdir, name * "_" * tag * "." * ext))
    end
    println("saved: ", joinpath(outdir, name * "_" * tag * ".png"))
    return Dc, eta_c
end

if abspath(PROGRAM_FILE) == @__FILE__
    name = length(ARGS) >= 1 ? ARGS[1] : "fidelity_p0.2"
    folder = length(ARGS) >= 2 ? ARGS[2] : "save_results/xxz"
    plot_crossing(name; folder = folder)
end
