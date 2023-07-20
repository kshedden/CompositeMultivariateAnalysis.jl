function gendat2(rng, n)
    p, q = 7, 6
    X = randn(rng, n, p)

    # Groups
    ng = 10
    Xg = sample(rng, 1:ng, n)

    Mx = randn(rng, p, ng)

    r = 0.5
    for j in 2:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:,j]
    end

    # Group effects
    for i in 1:n
        X[i, :] += Mx[:, Xg[i]]
    end

    return X, Xg
end

@testset "reshape parameters (one-sided)" begin

    p = 4
    d = 2
    m = p*d

    v = collect(1.:m)
    A = CompositeMultivariateAnalysis.unpack(v, p, d)

    A0 = [1. 5; 2 6; 3 7; 4 8]

    @test isapprox(A, A0)

    # Check that we have a view
    A[2, 1] = 99
    @test v[2] == 99
end

@testset "between-group covariance" begin

    # Number of groups
    ngrp = 200

    # Total sample size
    n = 5000

    # Number of variables
    p = 3

    # Autocorrelation parameter
    r = 0.5

    rng = StableRNG(123)
    nrep = 5
    cfg = MVAconfig(pca=0.0, lda=1.0, core=0.0)
    B = zeros(p, p)
    W = zeros(p, p)
    for k in 1:nrep
        X = Matrix{Union{Missing,Float64}}(randn(rng, n, p))
        for j in 2:p
           X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:, j]
        end
        M = randn(rng, p, ngrp)
        G = rand(1:ngrp, n)
        for i in 1:n
            X[i, :] .+= M[:, G[i]]
        end

        # Introduce some missing values
        for i in 1:n, j in 1:p
            if rand(rng) < 0.05
                X[i, j] = missing
            end
        end
        X[G .== 1, 1] .= missing
        X[G .== 2, 2] .= missing

        m = fit(MVA, X, G; d=3, config=cfg, dofit=false)
        W .+= m.Smw
        B .+= m.Sb
    end

    B /= nrep # Estimated between-group covariance
    W /= nrep # Estimated within-group covariance

    # Actual between-group and within-group covariances
    B0 = I(p)
    W0 = [r^abs(i-j) for i in 1:p, j in 1:p]

    @test isapprox(W, W0, atol=0.1, rtol=0.05)
    @test isapprox(B, B0, atol=0.1, rtol=0.1)
end

@testset "within-group covariance" begin

    # Number of groups
    ngrp = 4

    # Total sample size
    n = 5000

    # Number of variables
    p = 3

    # Autocorrelation parameters
    r = [0.8^k for k in 1:ngrp]

    rng = StableRNG(123)
    X = randn(rng, n, p)
    G = sample(rng, 1:ngrp, n)
    for k in 1:ngrp
        jj = G .== k
        for j in 2:p
            X[jj, j] = r[k]*X[jj, j-1] + sqrt(1 - r[k]^2)*X[jj, j]
        end
    end

    W, pw = CompositeMultivariateAnalysis.within_cov(X, G)
    @test length(W) == ngrp

    for k in 1:ngrp
        M = [r[k]^abs(i-j) for i in 1:p, j in 1:p]
        @test abs(tr(M \ W[k]) - p) < 0.2
    end
end

@testset "gradient (one-sided)" begin

    n = 1000
    d = 2
    rho = 0.0
    rng = StableRNG(123)
    vi = [0., 1., 2.]

    for k in 1:10
        X, G = gendat2(rng, n)
        p = size(X, 2)
        r = p*d
        v = randn(rng, r)
        z = randn(rng, r)
        for (pca, lda, core) in Iterators.product(vi, vi, vi)
            if pca + lda + core == 0
                continue
            end
            cfg = MVAconfig(pca=pca, lda=lda, core=core)
            m = fit(MVA, X, G; d=d, config=cfg, dofit=false)

            ag = zeros(r)
            v0 = copy(v)
            CompositeMultivariateAnalysis.admm_grad!(m, v, ag, z, rho)
            f = x -> CompositeMultivariateAnalysis.admm_objective(m, v, z, rho)
            ng = grad(central_fdm(5, 1), f, v)[1]
            @test isapprox(ag, ng)
            @assert isapprox(ag, ng)
        end
    end
end

@testset "test pca" begin

    n = 1000
    rng = StableRNG(123)
    cfg = MVAconfig(lda=0., pca=1., core=0.)

    for d in [1, 2, 3]
        for k in 1:10
            X, G = gendat2(rng, n)
            p = size(X, 2)
            m = fit(MVA, X, G; d=d, config=cfg)
            A = coef(m)
            v = CompositeMultivariateAnalysis.fit_pca(m)
            A0 = CompositeMultivariateAnalysis.unpack(v, p, d)
            @test candist(A, A0) >= 0.99*d
        end
    end
end

@testset "test lda" begin

    n = 1000
    rng = StableRNG(123)
    cfg = MVAconfig(lda=1., pca=0.1, core=0.)

    for d in [1, 2, 3]
        for k in 1:10
            X, G = gendat2(rng, n)
            p = size(X, 2)

            # The results of LDA
            m = fit(MVA, X, G; d=d, config=cfg, dofit=false)
            A0 = CompositeMultivariateAnalysis.fit_lda(m)
            A0 = CompositeMultivariateAnalysis.unpack(A0, p, d)

            m = fit(MVA, X, G; d=d, config=cfg)
            A1 = coef(m)

            @test candist(A0, A1) >= 0.99*d
        end
    end
end

@testset "test rotate" begin

    n = 1000
    d = 3
    rng = StableRNG(123)
    cfg = MVAconfig(lda=1., pca=0.1, core=0.)

    X, G = gendat2(rng, n)
    m = fit(MVA, X, G; d=d, config=cfg)
    CompositeMultivariateAnalysis.rotate!(m, method=:lda)
    A = coef(m)
    B = A' * m.Sx * A
    @test issorted(diag(B), rev=true)
    B .*= (1 .- I(d))
    @test isapprox(B, zeros(d, d), atol=1e-8, rtol=1e-8)
end
