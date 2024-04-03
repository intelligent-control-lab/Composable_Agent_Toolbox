import math
import ortoolpy

class Knapsack:

    def __init__(self, sim, infra, C, beta):
        self.sim = sim
        self.infra = infra
        self.C = C
        self.beta = beta

    # KL-divergence for two gaussians (mu, sigma^2)
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    def _kl_gauss(self, f, g):
        return 0.5 * (math.log(g[1] / f[1]) + (f[1] + (f[0] - g[0])**2) / g[1] - 1)

    # assume they're in the same lane
    def _value(self, sender, receiver, subject):
        alpha = 1 # TODO: tune params
        beta = 1
        gamma = 1
        d = 1 / (self.sim.x['pR'][receiver] - self.sim.x['pH'][subject])**2
        # dv = 
        post_p, post_v = self.infra.test_share(sender, receiver, subject)
        kl_p = self._kl_gauss(post_p, self.infra.bel[receiver][subject]['pos'])
        kl_v = self._kl_gauss(post_v, self.infra.bel[receiver][subject]['vel'])
        return alpha * d + beta * kl_p + gamma * kl_v

    def sack(self, comms):
        v = [self._value(a, b, h) for (a, b, h) in comms]
        c = [self.C[a][b] for (a, b, _) in comms]
        return ortoolpy.knapsack(v, c, self.beta)
