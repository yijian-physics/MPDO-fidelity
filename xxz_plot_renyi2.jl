## Plot the Renyi-2 correlator scan produced by
##     julia xxz_init_data.jl <p> renyi2
## in the style of Fig. 3(a)-(b) of 2503.14221 [Guo]: L^0.66 * C_X^II(L/4, 3L/4)
## against the anisotropy Delta, one curve per system size.
##
##     julia xxz_plot_renyi2.jl [name] [folder]
## default name = renyi2_p0.2, folder = save_results/xxz

using Plots
using LaTeXStrings
using Printf

function read_array_json(path::AbstractString)
    ## Reader for the dependency-free format written by save_array_json.
    txt = read(path, String)
    numvec(s) = [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
    grab(key) = numvec(match(Regex("\"$key\"\\s*:\\s*\\[([^\\]]*)\\]"), txt).captures[1])
    p = parse(Float64, strip(match(r"\"p\"\s*:\s*([^,\n]+)", txt).captures[1]))
    N_tot = Int.(grab("N_tot"))
    Delta_tot = grab("Delta_tot")
    body = match(r"\"data\"\s*:\s*\[(.*?)\n\s*\]"s, txt).captures[1]
    rows = [numvec(m.captures[1]) for m in eachmatch(r"\[([^\]]*)\]", body)]
    A = permutedims(reduce(hcat, rows))
    return A, N_tot, Delta_tot, p
end

function plot_renyi2(name="renyi2_p0.2"; folder="save_results/xxz", outdir="figures",
                     eta=0.66, ysym=raw"C_X^{\mathrm{II}}(L/4,3L/4)", ttl=nothing)
    ## Works for any observable saved in this format - pass ysym/eta to plot the
    ## Uhlmann fidelity scan instead of the Renyi-2 correlator.
    A, N_tot, Delta_tot, p = read_array_json(joinpath(folder, name * ".json"))
    mkpath(outdir)

    ## Fig. 3(a) plots L^0.66 * C. At p = 0.1 the Renyi-2 correlator decays as
    ## L^{-0.66} exactly on the critical line, so this combination is L-independent
    ## at Delta_c and fans out on either side: the curves for different L crossing
    ## at a single point IS the transition. (0.66 is the exponent at this p only -
    ## along the critical line it decreases with increasing decoherence.)
    marks = [:circle, :square, :utriangle, :diamond, :star5, :hexagon]
    ## build the label as one LaTeX string - interpolating a LaTeXString into
    ## another with %$() leaves the escapes unrendered
    ylab = eta == 0 ? latexstring(ysym) : latexstring("L^{$(eta)}\\, $(ysym)")
    mkplot(xlab, t) = plot(xlabel = xlab, ylabel = ylab, title = t,
                           legend = :topleft, framestyle = :box, size = (620, 470),
                           titlefontsize = 10, guidefontsize = 11, legendfontsize = 9)

    bc = occursin("obc", lowercase(name)) ? "OBC" : "PBC"
    base = ttl === nothing ? "XXZ under XX decoherence, p = $(p/2), $bc" : ttl
    plt = mkplot(L"\Delta", base)
    plt2 = mkplot(L"\Delta \log L", base * " -- collapse")
    for (ii, N) in enumerate(N_tot)
        y = A[ii, :] .* N^eta
        keep = .!isnan.(y)
        any(keep) || continue
        m = marks[mod1(ii, length(marks))]
        plot!(plt,  Delta_tot[keep],            y[keep], label = L"L = %$(N)", marker = m, markersize = 4, lw = 1.6)
        plot!(plt2, Delta_tot[keep] .* log(N),  y[keep], label = L"L = %$(N)", marker = m, markersize = 4, lw = 1.6)
    end
    vline!(plt, [0.0], ls = :dash, lc = :black, alpha = 0.6, label = L"\Delta_c = 0")
    vline!(plt2, [0.0], ls = :dash, lc = :black, alpha = 0.6, label = "")

    for (pl, tag) in ((plt, "fig3a"), (plt2, "fig3b"))
        for ext in ("png", "pdf")
            savefig(pl, joinpath(outdir, name * "_" * tag * "." * ext))
        end
        println("saved: ", joinpath(outdir, name * "_" * tag * ".png"))
    end
    return plt, plt2
end

function renyi2_exponents(name="renyi2_p0.2"; folder="save_results/xxz")
    ## Effective decay exponent eta(Delta) from a least-squares fit of log C against
    ## log L over the available sizes. On the critical line this is the exponent the
    ## paper's y-axis divides out (0.66 at p = 0.1).
    A, N_tot, Delta_tot, p = read_array_json(joinpath(folder, name * ".json"))
    x = log.(float.(N_tot))
    println("Delta     eta_eff")
    for (jj, D) in enumerate(Delta_tot)
        y = log.(A[:, jj])
        keep = .!isnan.(y)
        sum(keep) >= 2 || continue
        xs, ys = x[keep], y[keep]
        slope = sum((xs .- sum(xs)/length(xs)) .* ys) / sum((xs .- sum(xs)/length(xs)).^2)
        @printf("%6.3f   %7.4f\n", D, -slope)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    name = length(ARGS) >= 1 ? ARGS[1] : "renyi2_p0.2"
    folder = length(ARGS) >= 2 ? ARGS[2] : "save_results/xxz"
    plot_renyi2(name; folder = folder)
end
