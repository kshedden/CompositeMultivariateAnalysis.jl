using LinearAlgebra
using Statistics
using Optim
using NaNStatistics
using DataFrames

"""
     BiMVAconfig

Specify weights for each component of the objective function.
"""
struct BiMVAconfig
    cca::Float64
    pcax::Float64
    pcay::Float64
end

function BiMVAconfig(; cca=1, pcax=0, pcay=0)
    if min(cca, pcax, pcay) < 0
        error("BiMVA configuration weights must be non-negative.")
    elseif max(cca, pcax, pcay) == 0
        error("At least one component of the BiMVA objective must have positive weight")
    end
    return BiMVAconfig(cca, pcax, pcay)
end

"""
     BiMVA

A composite of canonical correlation analysis (CCA) and principal component
analysis (PCA).  Two blocks of variables are analyzed, assessing the correlations
between them and the variances within each block.
"""
mutable struct BiMVA

    # The raw data for the first variable (variables are columns, observations are rows)
    X::AbstractMatrix

    # The centered data for the first variable
    Xc::AbstractMatrix

    # The mean of the first variable
    Xm::AbstractVector

    # The raw data for the second variable
    Y::AbstractMatrix

    # The centered data for the second variable
    Yc::AbstractMatrix

    # The mean of the second variable
    Ym::AbstractVector

    # The cross-covariance between the first and second variables
    Sxy::AbstractMatrix

    # The covariance of the first variable
    Sxx::AbstractMatrix

    # The covariance of the second variable
    Syy::AbstractMatrix

    # The number of factors to extract
    d::Int

    # Weights for each component of the loss function
    cfg::BiMVAconfig

    # The estimated coefficients for X
    A::Matrix{Float64}

    # The estimated coefficients for Y
    B::Matrix{Float64}

    # Did the final ADMM iteration converge?
    converged::Bool
end

function center(X::AbstractMatrix)
    mn = [mean(skipmissing(v)) for v in eachcol(X)]
    Xc = copy(X)
    for j in eachindex(mn)
        # Propagate NaN's
        Xc[:, j] .-= mn[j]
    end
    return Xc, mn
end

function unpack(v::Vector{Float64}, p::Int, q::Int, d::Int)
    A = reshape(@view(v[1:p*d]), p, d)
    B = reshape(@view(v[p*d+1:end]), q, d)
    return A, B
end

function objective(m::BiMVA, v::Vector{Float64}; ridge::Float64=1e-4)

    (; Sxy, Sxx, Syy, d, cfg) = m
    (; cca, pcax, pcay) = cfg

    p, q = size(Sxy)
    A, B = unpack(v, p, q, d)

    f = 0.0

    # CCA
    qxy = A' * Sxy * B
    qxx = Symmetric(A' * Sxx * A) + ridge*I(d)
    qyy = Symmetric(B' * Syy * B) + ridge*I(d)
    f += cca * (logabsdet(qxy)[1] - 0.5*logdet(qxx) - 0.5*logdet(qyy))

    # PCA
    f += pcax * (logdet(qxx) - logdet(A' * A))
    f += pcay * (logdet(qyy) - logdet(B' * B))

    return f
end

function admm_objective(m::BiMVA, v::Vector{Float64}, z::Vector{Float64},
                        rho::Float64; ridge::Float64=1e-4)
    return objective(m, v; ridge=ridge) - rho * sum(abs2, v - z)
end

function grad!(m::BiMVA, v::Vector{Float64}, g::Vector{Float64}; ridge::Float64=1e-4)

    (; Sxy, Sxx, Syy, d, cfg) = m
    (; cca, pcax, pcay) = cfg

    p, q = size(Sxy)
    A, B = unpack(v, p, q, d)
    g .= 0
    Ga, Gb = unpack(g, p, q, d)

    # CCA
    qxy = A' * Sxy * B
    qxx = Symmetric(A' * Sxx * A) + ridge*I(d)
    qyy = Symmetric(B' * Syy * B) + ridge*I(d)
    Ga .+= cca * ((Sxy * B) / qxy - Sxx * A / qxx)
    Gb .+= cca * ((Sxy' * A) / qxy' - Syy * B / qyy)

    # PCA
    Ga .+= 2 * pcax * (Sxx * A / qxx - A / (A' * A))
    Gb .+= 2 * pcay * (Syy * B / qyy - B / (B' * B))
end

function admm_grad!(m::BiMVA, v::Vector{Float64}, g::Vector{Float64},
                    z::Vector{Float64}, rho::Float64; ridge::Float64=1e-4)
    grad!(m, v, g; ridge=ridge)
    g .-= 2 * rho * (v - z)
end

function fit(::Type{BiMVA}, X, Y; d::Int=1, config=BiMVAconfig(), rho_max::Float64=100.0,
             rho_ratio::Float64=1.2, dofit::Bool=true, ridge::Float64=1e-4,
             verbose::Bool=false)

    n = size(X, 1)
    if n != size(Y, 1)
        error("X and Y must have the same number of rows")
    end

    XY = hcat(X, Y)
    XY = coalesce.(XY, NaN)
    XY = disallowmissing(XY)
    C = nancov(XY)
    p, q = size(X, 2), size(Y, 2)
    Sxy = C[1:p, p+1:end]
    Sxx = Symmetric(C[1:p, 1:p])
    Syy = Symmetric(C[p+1:end, p+1:end])

    Xc, Xm = center(X)
    Yc, Ym = center(Y)

    cca = BiMVA(X, Xc, Xm, Y, Yc, Ym, Sxy, Sxx, Syy, d, config,
                zeros(0, 0), zeros(0, 0), false)
    if !dofit
        return cca
    end

    v = fit_pca(cca)
    v = fit_admm(cca, v; rho_max=rho_max, rho_ratio=rho_ratio, verbose=verbose, ridge=ridge)

    p = size(X, 2)
    q = size(Y, 2)
    A, B = unpack(v, p, q, d)
    cca.A = copy(A)
    cca.B = copy(B)
    scale!(cca)

    return cca
end

function coef(cca::BiMVA)
    return (cca.A, cca.B)
end

# Return the symmetric square root of A, which must be symmetric.
function ssqrt(A)
    a,b = eigen(Symmetric(A))
    if minimum(a) < -1e-8
        @warn("Minimum eigenvalue: $(minimum(a))")
    end
    a = clamp.(a, 0, Inf)
    return Symmetric(b * Diagonal(sqrt.(a)) * b')
end

function fit_cca(cca::BiMVA)
    (; Sxy, Sxx, Syy) = cca
    M = ssqrt(Sxx) \ Sxy / ssqrt(Syy)
    u, s, v = svd(M)
    u = ssqrt(Sxx) \ u
    v = ssqrt(Syy) \ v
    return u, v
end

function fit_pca(cca::BiMVA)

    (; d, Sxx, Syy) = cca

    p = size(Sxx, 1)
    q = size(Syy, 1)
    r = p*d + q*d
    v = zeros(r)
    A, B = unpack(v, p, q, d)

    a, b = eigen(Symmetric(Sxx))
    ii = sortperm(a, rev=true)
    A .= b[:, ii[1:d]]

    a, c = eigen(Symmetric(Syy))
    ii = sortperm(a, rev=true)
    B .= c[:, ii[1:d]]

    return v
end

function admm_center!(cca::BiMVA, v::Vector{Float64}, z::Vector{Float64})

    (; d, Sxx, Syy) = cca

    p = size(Sxx, 1)
    q = size(Syy, 1)

    A, B = unpack(v, p, q, d)

    Az, Bz = unpack(z, p, q, d)
    Az .= Matrix(qr(A).Q)
    Bz .= Matrix(qr(B).Q)
end

function fit_admm(mva::BiMVA, v::Vector{Float64}; rho_max::Float64=10.0,
                  rho_ratio::Float64=1.2, verbose::Bool=false, ridge::Float64=1e-4)

    (; Sxx, Syy, d) = mva

    p = size(Sxx, 1)
    q = size(Syy, 1)

    rho = 1.0
    z = similar(v)
    rr = nothing
    while rho < rho_max
        verbose && print("rho=$(rho)...")
        admm_center!(mva, v, z)
        f = v -> -admm_objective(mva, v, z, rho; ridge=ridge)
        g! = function(gr, v)
            admm_grad!(mva, v, gr, z, rho; ridge=ridge)
            gr .*= -1
        end

        rr = optimize(f, g!, v, LBFGS())
        v = Optim.minimizer(rr)
        mva.converged = Optim.converged(rr)
        verbose && println(mva.converged)

        rho *= rho_ratio
    end

    return v
end

function escore(X, Sx, A; verbose=false)
    n, p = size(X)
    d = size(A, 2)
    iobs = zeros(Bool, p)
    scores = Matrix{Union{Float64,Missing}}(missing, n, d)
    SxA = Sx * A
    for i in 1:n
        iobs .= .!ismissing.(X[i, :])
        if verbose && (i % 10 == 0)
            println("$(i)/$(n)")
        end
        if any(iobs)
            x = disallowmissing(X[i, iobs])
            scores[i, :] .= SxA[iobs, :]' * (Symmetric(Sx[iobs, iobs]) \ x)
        end
    end

    return scores
end

function predict(mva::BiMVA; verbose=false)
    (; A, B, Sxx, Syy, Xc, Yc) = mva

    p = size(Xc, 2)
    q = size(Yc, 2)

    n = size(Xc, 1)
    scorex = escore(Xc, Sxx, A; verbose=verbose)
    scorey = escore(Yc, Syy, B; verbose=verbose)

    return (scorex, scorey)
end

function rotate_cor!(mva::BiMVA)

    (; A, B, Sxy, Sxx, Syy) = mva

    Sxy = A' * Sxy * B
    Sxx = Symmetric(A' * Sxx * A)
    Syy = Symmetric(B' * Syy * B)

    Rx = ssqrt(Sxx)
    Ry = ssqrt(Syy)

    M = Rx \ Sxy / Ry
    u, s, v = svd(M)

    mva.A = A * (Rx \ u)
    mva.B = B * (Ry \ v)
end

function scale!(mva::BiMVA)
    (; Sxy, d, A, B) = mva
    p, q = size(Sxy)
    for j in 1:d
        A[:, j] ./= norm(A[:, j])
        B[:, j] ./= norm(B[:, j])
    end
end

function rotate!(mva::BiMVA, method=:cor)
    if method == :cor
        rotate_cor!(mva)
    else
        error("Unknown rotation method '$(method)'")
    end
    scale!(mva)
end

function cor(mva::BiMVA)
    (; A, B, Sxy, Sxx, Syy) = mva

    sx = sqrt.(diag(A' * Sxx * A))
    sy = sqrt.(diag(B' * Syy * B))
    return diag(Diagonal(1 ./ sx) * A' * Sxy * B * Diagonal(1 ./ sy))
end
