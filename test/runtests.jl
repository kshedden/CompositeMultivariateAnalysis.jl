using CompositeMultivariateAnalysis
using Test
using FiniteDifferences
using LinearAlgebra
using Statistics
using StatsBase
using StableRNGs
using NaNStatistics

function candist(A, B)
    A, _, _ = svd(A)
    B, _, _ = svd(B)
    _, s, _ = svd(A' * B)
    return sum(s)
end

#include("onesided.jl")
include("twosided.jl")

