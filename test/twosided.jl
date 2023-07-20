using Statistics

function gendat2(rng, n)
    p, q = 7, 6
    X = randn(rng, n, p)
    Y = randn(rng, n, q)
    r = 0.5
    for j in 2:p
        X[:, j] = r*X[:, j-1] + sqrt(1-r^2)*X[:,j]
    end
    Y[:, 6] = r*Y[:, 5] + sqrt(1-r^2)*Y[:, 6]
    Y[:, 1] += X[:, 1]
    Y[:, 2] += X[:, 3] + X[:, 4]
    return X, Y
end

@testset "reshape parameters (two-sided)" begin

    p = 4
    q = 3
    d = 2
    m = p*d + q*d

    v = collect(1.:m)
    A, B = CompositeMultivariateAnalysis.unpack(v, p, q, d)

    A0 = [1. 5; 2 6; 3 7; 4 8]
    B0 = [9. 12; 10 13; 11 14]

    @test isapprox(A, A0)
    @test isapprox(B, B0)

    # Check that we have a view
    A[2, 1] = 99
    @test v[2] == 99
    @test isapprox(B, B0)
    A[2, 1] = A0[2, 1]

    B[1, 2] = 99
    @test v[12] == 99
    @test isapprox(A, A0)
end

@testset "BiMVA invariance" begin

    n = 1000
    d = 2
    rng = StableRNG(123)
    vi = [0., 1., 2.]

    for k in 1:10
        X, Y = gendat2(rng, n)
        p, q = size(X, 2), size(Y, 2)
        r = p*d + q*d
        for (cca, pcax, pcay) in Iterators.product(vi, vi, vi)
            if cca + pcax + pcay == 0
                continue
            end
            cfg = BiMVAconfig(cca=cca, pcax=pcax, pcay=pcay)
            m = fit(BiMVA, Y, X; d=d, config=cfg, dofit=false)
            v = randn(r)
            f = CompositeMultivariateAnalysis.objective(m, v; ridge=0.)
            f2 = CompositeMultivariateAnalysis.objective(m, 2*v; ridge=0.)
            @test isapprox(f, f2)
        end
    end
end

@testset "gradient (two-sided)" begin

    n = 1000
    d = 2
    rho = 0.0
    rng = StableRNG(123)
    vi = [0., 1., 2.]

    for k in 1:10
        X, Y = gendat2(rng, n)
        p, q = size(X, 2), size(Y, 2)
        r = p*d + q*d
        v = randn(rng, r)
        z = randn(rng, r)
        for (cca, pcax, pcay) in Iterators.product(vi, vi, vi)
            if cca + pcax + pcay == 0
                continue
            end
            cfg = BiMVAconfig(cca=cca, pcax=pcax, pcay=pcay)
            m = fit(BiMVA, Y, X; d=d, config=cfg, dofit=false)

            ag = zeros(r)
            v0 = copy(v)
            CompositeMultivariateAnalysis.admm_grad!(m, v, ag, z, rho)
            f = x -> CompositeMultivariateAnalysis.admm_objective(m, v, z, rho)
            ng = grad(central_fdm(5, 1), f, v)[1]
            @test isapprox(ag, ng)
        end
    end
end

@testset "test pca" begin

    n = 1000
    rng = StableRNG(123)
    cfg = BiMVAconfig(cca=0., pcax=1., pcay=1.0)

    for d in [1, 2, 3]
        for k in 1:10
            X, Y = gendat2(rng, n)
            p, q = size(X, 2), size(Y, 2)
            r = p*d + q*d
            m = fit(BiMVA, X, Y; d=d, config=cfg)
            A, B = coef(m)
            v = CompositeMultivariateAnalysis.fit_pca(m)
            A0, B0 = CompositeMultivariateAnalysis.unpack(v, p, q, d)
            @test candist(A, A0[:, 1:d]) >= 0.99*d
            @test candist(B, B0[:, 1:d]) >= 0.99*d
        end
    end
end

@testset "test cca" begin

    n = 1000
    rng = StableRNG(123)
    cfg = BiMVAconfig(cca=1., pcax=0., pcay=0.)

    for d in [1,]
        for k in 1:10
            X, Y = gendat2(rng, n)
            p, q = size(X, 2), size(Y, 2)
            r = p*d + q*d
            m = fit(BiMVA, X, Y; d=d, config=cfg)
            A, B = coef(m)
            Sxy, Sxx, Syy = m.Sxy, m.Sxx, m.Syy
            u, v = CompositeMultivariateAnalysis.fit_cca(m)

            @test candist(u, A) >= 0.97*d
            @test candist(v, B) >= 0.97*d
        end
    end
end

@testset "fit" begin

    n = 1000
    d = 2
    rng = StableRNG(123)
    cfg = BiMVAconfig(cca=1., pcax=0.1, pcay=0.1)

    A0 = [0 1; 0 0; 1 0; 1 0; 0 0; 0 0; 0 0]
    B0 = [0 1; 1 0; 0 0; 0 0; 0 0; 0 0]

    r1, r2 = 0.0, 0.0
    da, db = 0.0, 0.0
    nrep = 100
    for k in 1:nrep
        X, Y = gendat2(rng, n)
        p, q = size(X, 2), size(Y, 2)
        r = p*d + q*d
        m = fit(BiMVA, X, Y; d=d, config=cfg)
        A, B = coef(m)
        Qx, Qy = predict(m)

        # Check that A and B are approximately orthogonal
        @test norm(A'*A - I(d)) < 0.01
        @test norm(B'*B - I(d)) < 0.01

        da += candist(A, A0)
        db += candist(B, B0)

        r1 += abs(cor(Qx[:, 1], Qy[:, 1]))
        r2 += abs(cor(Qx[:, 2], Qy[:, 2]))
    end

    r1 /= nrep
    r2 /= nrep
    da /= nrep
    db /= nrep

    r1_expected = 2 / sqrt(6)
    r2_expected = 1 / sqrt(2)

    @test isapprox(r1, r1_expected, atol=0.05)
    @test isapprox(r2, r2_expected, atol=0.05)

    @test da >= 1.9
    @test db >= 1.9
end

