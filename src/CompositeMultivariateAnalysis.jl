module CompositeMultivariateAnalysis

import StatsAPI: fit, coef, predict
import Statistics: cor

export MVA, MVAconfig, BiMVA, BiMVAconfig, rotate!, cor

export fit, coef, predict

include("onesided.jl")
include("twosided.jl")

end
