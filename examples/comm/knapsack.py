import math

# KL-divergence for two gaussians
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
def KL_gauss(f, g):
    return math.log(g[1] / f[1]) + (f[1]**2 (f[0] - g[0])**2) / (2 * g[1]**2) - 0.5

def alg():
    pass