import random

class Infrastructure:

    def __init__(self, sim):
        self.sim = sim
        self.obs = self._init_obs()
        self.bel = self._init_bel()

    def _init_obs(self):
        return [[{'pos': [], 'vel': []} for _ in range(self.sim.m)] 
                    for _ in range(self.sim.n)]

    def _init_bel(self):
        s0 = (1, 1) # TODO: tune params
        temp = [[] for _ in range(self.sim.n)]
        for i in range(self.sim.n):
            for j in range(self.sim.m):
                p, v = self.sense(i, j)
                temp[i].append({'pos': (p, s0[0]), 
                                'vel': (v, s0[1])})
        return temp

    # from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    # bayesian inference for gaussian prior and likelihood
    def _bayes_gauss(self, prior, lhood, obs):
        sigma2 = 1 / (len(obs) / lhood[1] + 1 / prior[1]) # eq. (20)
        mu = sigma2 * (prior[0] / prior[1] + sum(obs) / lhood[1]) # eq. (24)
        return (mu, sigma2)
    
    def _update_belief(self, sender, receiver, subject, update=True):
        # TODO: figure out whether we're actually estimating this at every timestep (hence dt)
        pos_est = self.bel[sender][subject]['pos'][0] + self.bel['vel'][sender][0] * self.sim.dt # integrate to guess
        vel_est = self.bel[sender][subject]['vel'][0] # constant velocity assumption

        sigma = (1, 1) # TODO: tune params
        # uncertainty of estimate is proportional to uncertainty of observation [assume (mean, stdev^2) tuples]
        lhood_p = (pos_est, sigma[0]**2 * self.bel[sender][subject]['pos'][1]) # (mean, stdev^2)
        lhood_v = (vel_est, sigma[1]**2 * self.bel[sender][subject]['vel'][1])

        prior_p = self.bel[receiver][subject]['pos'] # (mean, stdev^2)
        prior_v = self.bel[receiver][subject]['vel']

        obs_p = self.obs[sender][subject]['pos'] # list of all observations
        obs_v = self.obs[sender][subject]['vel']

        post_p = self._bayes_gauss(prior_p, lhood_p, obs_p) # (mean, stdev^2)
        post_v = self._bayes_gauss(prior_v, lhood_v, obs_v)

        if update:
            self.bel[receiver][subject]['pos'] = post_p
            self.bel[receiver][subject]['vel'] = post_v

        return (post_p, post_v)


    def sense(self, observer, subject):
        pos = self.sim.x['pH'][subject]
        vel = self.sim.x['vH'][subject]
        dist = abs(pos - self.sim.x['pR'][observer])

        alpha = (0.01, 0.01) # TODO: tune params
        pos += random.gauss(0, alpha[0]*dist**2 + alpha[1]*pos)
        vel += random.gauss(0, alpha[0]*dist**2 + alpha[1]*vel)

        self.obs[observer][subject]['pos'].append(pos)
        self.obs[observer][subject]['vel'].append(vel)

        self._update_belief(observer, observer, subject)
        
        return (pos, vel)
    
    def share(self, sender, receiver, subject):
        self._update_belief(sender, receiver, subject)

    def test_share(self, sender, receiver, subject):
        return self._update_belief(sender, receiver, subject, update=False)