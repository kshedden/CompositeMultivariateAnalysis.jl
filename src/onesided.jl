using LinearAlgebra
using Statistics
using Optim
using NaNStatistics
using DataFrames

"""
    MVAconfig

Specify weights for each component of the objective function.
"""
struct MVAconfig
    pca::Float64
    lda::Float64
    core::Float64
end

function MVAconfig(; pca=0, lda=0, core=0)
    if min(pca, lda, core) < 0
        error("MVA configuration weights must be non-negative.")
    elseif max(pca, lda, core) == 0
        error("At least one component of the MVA objective must have positive weight")
    end
    return MVAconfig(pca, lda, core)
end

"""
    MVA

A composite of linear discriminant analysis (LDA), principal component
analysis (PCA), and covariance reduction (CORE).
"""
mutable struct MVA

    # The raw data (variables are columns, observations are rows)
    X::AbstractMatrix

    # Group labels
    Gr::AbstractVector

    # The centered data
    Xc::AbstractMatrix

    # The mean of each variable
    Xm::AbstractVector

    # The covariance matrix
    Sx::AbstractMatrix

    # The mean between-group covariance
    Sb::AbstractMatrix

    # The mean within-group covariance
    Smw::AbstractMatrix

    # The within-group covariance for each group
    Sw::Vector{AbstractMatrix}

    # The number of factors to extract
    d::Int

    # Weights for each component of the loss function
    cfg::MVAconfig

    # The estimated coefficients
    A::Matrix{Float64}

    # Proportion of data per group.
    pw::Vector{Float64}

    # Did the final ADMM iteration converge?
    converged::Bool
end

function unpack(v::Vector{Float64}, p::Int, d::Int)
    # Return a matrix that contains the loading patterns stored
    # in packed form in 'v'.  The returned matrix shares memory
    # with 'v'.
    A = reshape(@view(v[1:p*d]), p, d)
    return A
end

# Estimate the between-group covariance by subtracting the
# within-group covariance from the marginal covariance.
function between_cov(Sx, Smw)
    Sb = Sx - Smw
    a, b = eigen(Symmetric(Sb))
    if minimum(a) < -1e-8
        @warn("Between-group covariance minimum eigenvalue: $(minimum(a))")
    end
    a = clamp.(a, 0, Inf)
    return Symmetric(b * Diagonal(a) * b')
end

function objective(m::MVA, v::Vector{Float64}; ridge::Float64=1e-4)
    # Evaluate the objective function used to estimate the composite
    # MVA parameters.
    (; Sx, Sb, Smw, Sw, d, cfg, pw) = m
    (; pca, lda, core) = cfg

    p = size(Sx, 1)
    A = unpack(v, p, d)
    f = 0.0

    # LDA
    if lda > 0
        qb = Symmetric(A' * Sb * A)
        qw = Symmetric(A' * Smw * A + ridge*I(d))
        f += lda * (logdet(qb) - logdet(qw))
    end

    # PCA
    if pca > 0
        qx = Symmetric(A' * Sx * A)
        f += pca * (logdet(qx) - logdet(A' * A))
    end

    # CORE
    if core > 0
        qq = Symmetric(A' * Sx * A)
        f += core * logdet(qq)
        for j in eachindex(Sw)
            qq = Symmetric(A' * Sw[j] * A + ridge*I(d))
            f -= core * pw[j] * logdet(qq)
        end
    end

    return f
end

function admm_objective(m::MVA, v::Vector{Float64}, z::Vector{Float64}, rho::Float64;
                        ridge::Float64=1e-4)
    # The ADMM objective function is the objective function plus a quadratic
    # penalty that keeps the argument close to a given orthogonal matrix.
    return objective(m, v; ridge=ridge) - rho * sum(abs2, v - z)
end

function grad!(m::MVA, v::Vector{Float64}, g::Vector{Float64}; ridge::Float64=1e-4)
    # Calculate the gradient of the objective function at the point 'g',
    # and store the results in 'g'.
    (; Sx, Sb, Smw, Sw, pw, d, cfg) = m
    (; lda, pca, core) = cfg

    p = size(Sx, 1)
    A = unpack(v, p, d)
    g .= 0
    Ga = unpack(g, p, d)

    # LDA
    if lda > 0
        q1 = Symmetric(A' * Sb * A)
        q2 = Symmetric(A' * Smw * A) + ridge*I(d)
        Ga .+= 2 * lda * ((Sb * A) / q1 - (Smw * A) / q2)
    end

    # PCA
    if pca > 0
        q1 = Symmetric(A' * Sx * A)
        Ga .+= 2 * pca * (Sx * A / q1 - A / (A' * A))
    end

    # CORE
    if core > 0
        Ga .+= 2 * core * (Sx * A) / Symmetric(A' * Sx * A)
        for j in eachindex(Sw)
            Ga .-= 2 * core * pw[j] * (Sw[j] * A) / Symmetric(A' * Sw[j] * A + ridge*I(d))
        end
    end
end

function admm_grad!(m::MVA, v::Vector{Float64}, g::Vector{Float64},
                    z::Vector{Float64}, rho::Float64; ridge::Float64=1e-4)
    # The gradient of the ADMM objective function.
    grad!(m, v, g; ridge=ridge)
    g .-= 2 * rho * (v - z)
end

function within_cov_groups(Z, G)

    Z = coalesce.(Z, NaN)

    # Calculate the within-group covariance matrix for each
    # group, and the proportion of data in each group,
    # accommodating missing data.
    n, p = size(Z)

    # Sort so that groups are contiguous
    ii = sortperm(G)
    Z = Z[ii, :]
    G = G[ii]

    # Get intervals for groups
    ix = findall(G[1:end-1] .!= G[2:end])
    ix = vcat(1, ix, n)

    Sw, pw = [], Float64[]
    for j in 1:length(ix)-1
        zm = Z[ix[j]:ix[j+1], :]
        c = nancov(zm)
        if any(isnan.(c))
            println(typeof(c))
            println(size(c))
            println(typeof(zm))
            println(size(zm))
            display(c)
            display(zm)
            error("")
        end
        push!(Sw, Symmetric(c))
        push!(pw, ix[j+1] - ix[j] + 1)
    end

    pw = pw ./ sum(pw)

    return Sw, pw
end

function mean_within_cov(Sw, pw)
    Smw = similar(first(Sw))
    Smw .= 0
    for k in eachindex(pw)
        Smw += pw[k] * Sw[k]
    end
    return Smw
end

function between_cov_direct(Z, G)
    # Estimate Cov E[Z | G], accommodating missing values.
    # The columns of Z have been centered.
    if length(G) == 0
        return zeros(0, 0)
    elseif length(G) != size(Z, 1)
        error("group labels must have same length as data matrix")
    end

    # Map unique group labels to positions 1, 2, ...
    ix = Dict{eltype(G),Int}()
    for (i,g) in enumerate(unique(G))
        ix[g] = i
    end

    p = size(Z, 2)
    m = length(ix)
    N = zeros(Int, p, m) # sample size per variable/group
    M = zeros(p, m) # sums (then means) per variable/group
    qm = zeros(Bool, p)
    for (i,g) in enumerate(G)
        qm .= .!ismissing.(Z[i, :])
        j = ix[g]
        N[:, j] += qm
        M[qm, j] += Z[i, qm]
    end

    # Weight per group
    wgt = sum(N, dims=1)[:]
    wgt = wgt ./ sum(wgt)

    # Convert group sums to group means
    M ./= N
    M = replace(M, Inf=>0)
    M = replace(M, NaN=>0)
    M = M * Diagonal(sqrt.(wgt))
    C = Symmetric(M * M')
    a, b = eigen(C)
    a = clamp.(a, 0, Inf)
    return Symmetric(b * Diagonal(a) * b')
end

function _get_start(mva::MVA, start)
    (; Sx, d, cfg) = mva
    (; pca, lda, core) = cfg
    p = size(Sx, 1)

    if !isnothing(start)
        return start
    elseif core == 0 && pca == 0
        return fit_lda(mva)
    elseif core == 0 && lda == 0
        return fit_pca(mva)
    elseif core == 0
        # Retraction of the point between the LDA and PCA solutions
        A = fit_lda(mva)
        A = reshape(A, p, d)
        B = fit_pca(mva)
        B = reshape(B, p, d)
        t = lda + pca
        w1, w2 = lda / t, pca / t
        W = w1*A*A' + w2*B*B'
        a, b = eigen(Symmetric(W))
        ii = sortperm(a, rev=true)
        b = b[:, ii[1:d]]
        return vec(b)
    else
        # TODO: need better starting value here
        return fit_pca(mva)
    end
end

function fit(::Type{MVA}, X, G; d::Int=1, config=MVAconfig(), start=nothing,
             rho_max::Float64=100.0, rho_ratio::Float64=1.2, ridge::Float64=1e-4,
             verbose=false, dofit=true)

    n, p = size(X)
    (; pca, lda, core) = config

    XX = coalesce.(X, NaN)
    XX = disallowmissing(XX)
    Sx = nancov(XX)

    Xc, Xm = center(X)

    Sw, pw, Smw, Sb = if max(lda, core) > 0
        Sw, pw = within_cov_groups(Xc, G)
        Smw = mean_within_cov(Sw, pw)
        println("Smw=", Smw)
        println(length(Sw))
        Sb = between_cov(Sx, Smw)
        Sw, pw, Smw, Sb
    else
        zeros(0, 0), zeros(0), zeros(0, 0), zeros(0, 0)
    end

    mva = MVA(X, G, Xc, Xm, Sx, Sb, Smw, Sw, d, config,
              zeros(0, 0), pw, false)
    if !dofit
        return mva
    end

    v = _get_start(mva, start)
    v = fit_admm(mva, v; rho_max=rho_max, rho_ratio=rho_ratio, ridge=ridge,
                 verbose=verbose)

    p = size(X, 2)
    A = unpack(v, p, d)
    mva.A = copy(A)
    scale!(mva)

    return mva
end

function coef(mva::MVA)
    return mva.A
end

function admm_center!(mva::MVA, v::Vector{Float64}, z::Vector{Float64})
    # Each round of ADMM has a quadratic penalty term that keeps
    # the loading matrix close to the orthogonal matrix returned
    # by this function.
    (; d, Sx) = mva

    p = size(Sx, 1)
    A = unpack(v, p, d)

    Az = unpack(z, p, d)
    Az .= Matrix(qr(A).Q)
end

function fit_admm(mva::MVA, v::Vector{Float64}; rho_max::Float64=10.0,
                  rho_ratio::Float64=1.2, ridge::Float64=ridge, verbose=false)
    # Use ADMM optimization to optimize the objective function over
    # the Stiefel manifold of orthgonal bases.
    (; Sx, d) = mva

    p = size(Sx, 1)
    rho = 1.0
    z = similar(v)
    rr = nothing

    while rho < rho_max
        verbose && println("rho=$(rho)")
        admm_center!(mva, v, z)
        f = v -> -admm_objective(mva, v, z, rho; ridge=ridge)
        g! = function(gr, v)
            admm_grad!(mva, v, gr, z, rho; ridge=ridge)
            gr .*= -1
        end

        rr = optimize(f, g!, v, LBFGS(), Optim.Options(show_trace=verbose))
        v = Optim.minimizer(rr)
        mva.converged = Optim.converged(rr)
        verbose && println("Converged=", mva.converged)

        rho *= rho_ratio
    end

    return v
end

function predict(mva::MVA)
    (; Xc, Sx, A) = mva
    return escore(Xc, Sx, A)
end

function fit_pca(mva::MVA)
    # Estimate the loadings using principal components analysis (PCA).
    (; Sx, d) = mva
    p = size(Sx, 1)
    a, b = eigen(Symmetric(Sx))
    ii = sortperm(a, rev=true)
    v = zeros(p*d)
    A = unpack(v, p, d)
    A .= b[:, ii[1:d]]
    return v
end

function fit_lda(mva::MVA)
    # Estimate the loadings using linear discriminant analysis (LDA).
    (; Sx, Smw, Sb, d) = mva
    p = size(Sx, 1)
    R = ssqrt(Smw)
    B = R \ Sb / R
    a, b = eigen(Symmetric(B))
    ii = sortperm(a, rev=true)
    v = zeros(p*d)
    A = unpack(v, p, d)
    A .= R \ b[:, ii[1:d]]
    return v
end

function scale!(mva::MVA)
    # Scale the loadings so that each component has unit norm.
    (; d, A) = mva
    for j in 1:d
        A[:, j] ./= norm(A[:, j])
    end
end

function rotate_pca!(mva::MVA)
    # Rotate the loadings to sequentially optimize
    # the PCA objective.
    (; Sx, A, d) = mva
    Sx = Symmetric(A' * Sx * A)
    a, b = eigen(Symmetric(Sx))
    ii = sortperm(a, rev=true)
    a = a[ii]
    b = b[:, ii[1:d]]
    mva.A = A * b
    scale!(mva)
end

function rotate_lda!(mva::MVA)
    # Rotate the loadings to sequentially optimize
    # the LDA objective.
    (; Sb, Smw, Sx, A, d) = mva
    Sb = Symmetric(A' * Sb * A)
    Smw = Symmetric(A' * Smw * A)
    R = ssqrt(Smw)
    M = Symmetric(R \ Sb / R)
    a, b = eigen(M)
    ii = sortperm(a, rev=true)
    a = a[ii]
    b = b[:, ii[1:d]]
    mva.A = A * (R \ b)
    scale!(mva)
end

"""
    rotate!(mva; method=:lda)

Convert the factor loadings to sequentially optimize
either the LDA objective or the PCA objective.
"""
function rotate!(mva::MVA; method=:lda)
    if method == :lda
        rotate_lda!(mva)
    elseif method == :pca
        rotate_pca!(mva)
    else
        error("Unknown rotation method '$(method)'")
    end
    scale!(mva)
end
