import numpy as np
import ortoolpy

class Knapsack:

    def __init__(self, model, sim, infra, C, beta):
        self.model = model
        self.sim = sim
        self.infra = infra
        self.C = C
        self.beta = beta

    # KL-divergence for two gaussians (mu, sigma^2)
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    def _kl_gauss(self, f, g):
        return 0.5 * (np.log(g[1] / f[1]) + (f[1] + (f[0] - g[0])**2) / g[1] - 1)

    # assume they're in the same lane
    def _value(self, sender, receiver, subject):
        alpha = 1 # TODO: tune params
        beta = 1
        gamma = 1
        
        # inputs = None
        # xH = {'p': self.sim.bel[sender][subject]['pos'], 'v': self.sim.bel[sender][subject]['vel']}
        # xR = {'p': self.sim.x['pR'][receiver], 'v': self.sim.x['pR'][receiver], 
        #       'g': self.sim.x['gR'][receiver]}

        dm = self.model(self.infra.bel[sender][subject]['pos'], self.infra.bel[sender][subject]['vel'], 
                        self.sim.x['pR'][receiver], self.sim.x['pR'][receiver]) # decision-making value from model

        post_p, post_v = self.infra.test_share(sender, receiver, subject)
        kl_p = self._kl_gauss(self.infra.bel[receiver][subject]['pos'], post_p)
        kl_v = self._kl_gauss(self.infra.bel[receiver][subject]['vel'], post_v)

        return alpha * dm + beta * kl_p[0] + beta * kl_p[1] + gamma * kl_v[0] + gamma * kl_v[1]

    def sack(self, comms):
        c = [self.C[a][b] for (a, b, _) in comms]
        v = [self._value(a, b, h) for (a, b, h) in comms]
        # print(c, v, self.beta)
        return ortoolpy.knapsack(c, v, self.beta)
