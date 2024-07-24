import random

import numpy as np

class Infrastructure:

    def __init__(self, sim):
        self.sim = sim
        self.obs = [[{'pos': [], 'vel': []} for _ in range(self.sim.m)] 
                    for _ in range(self.sim.n)]
        self.bel = [[{'pos': (np.array([0, 0]), np.array([0, 0])), 
                      'vel': (np.array([0, 0]), np.array([0, 0])), 
                      't': 0} for _ in range(self.sim.m)] 
                    for _ in range(self.sim.n)]
        self._init_bel()

    def _init_bel(self):
        s0 = (10, 10) # TODO: tune params
        for i in range(self.sim.n):
            for j in range(self.sim.m):
                p, v = self.sense(i, j, initial=True)
                self.bel[i][j]['pos'] = (p, np.array([s0[0]**2, s0[0]**2]))
                self.bel[i][j]['vel'] = (v, np.array([s0[1]**2, s0[1]**2]))

    # from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    # bayesian inference for gaussian prior and likelihood
    def _bayes_gauss(self, prior, lhood, obs):
        sigma2 = 1 / (len(obs) / lhood[1] + 1 / prior[1]) # eq. (20)
        # mu = sigma2 * (prior[0] / prior[1] + sum(obs) / lhood[1]) # eq. (24)
        mu = sigma2 * (prior[0] / prior[1] + (len(obs) * obs[-1]) / lhood[1]) # eq. (24) [MODIFIED TO REMOVE AVG]
        return (mu, sigma2)

    def _update_belief(self, sender, receiver, subject, update=True, new_obs=True):

        # estimate current state from belief from last timestep
        vel_est = self.bel[receiver][subject]['vel'][0] # constant velocity assumption
        delta_t = self.sim.t - self.bel[receiver][subject]['t'] # time since last update
        pos_est = self.bel[receiver][subject]['pos'][0] + vel_est * delta_t # integrate to guess

        # prior is estimated state of receiver
        diff = self.sim.x['pR'][receiver] - self.bel[receiver][subject]['pos'][0]
        rec_dist2 = diff.T @ diff
        sigma0_v2 = 0.05 * rec_dist2 # TODO: tune param
        n = len(self.obs[receiver][subject]['pos'])
        # TODO: not sure if using dt below is correct, but using delta_t breaks it when delta_t=0
        prior_p = (pos_est, np.array([sigma0_v2 * n * self.sim.dt, sigma0_v2 * n * self.sim.dt])) # (mu, sigma^2)
        prior_v = (vel_est, np.array([sigma0_v2 * n * self.sim.dt, sigma0_v2 * n * self.sim.dt]))
        
        # likelihood is latest observation of sender
        diff = self.sim.x['pR'][sender] - self.bel[sender][subject]['pos'][0]
        sen_dist2 = diff.T @ diff
        sigmaN2 = (0.25 * sen_dist2, 0.25 * sen_dist2) # TODO: tune params
        lhood_p = (self.obs[sender][subject]['pos'][-1], np.array([sigmaN2[0], sigmaN2[0]])) # (mu, sigma^2)
        lhood_v = (self.obs[sender][subject]['vel'][-1], np.array(sigmaN2[1], sigmaN2[1])) 

        # compute posterior for receiver
        post_p = self._bayes_gauss(prior_p, lhood_p, self.obs[sender][subject]['pos']) # (mu, sigma^2)
        post_v = self._bayes_gauss(prior_v, lhood_v, self.obs[sender][subject]['vel'])

        if update:
            self.bel[receiver][subject]['pos'] = post_p
            self.bel[receiver][subject]['vel'] = post_v
            self.bel[receiver][subject]['t'] = self.sim.t

        return (post_p, post_v)
    
    def _occluded(self, observer, subject):
        ccw = lambda a,b,c : (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        intersect = lambda a,b,c,d : ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)
        pH = self.sim.x['pH'][subject]
        pR = self.sim.x['pR'][observer]
        for w1, w2 in self.sim.x['walls']:
            if intersect(pH, pR, w1, w2): # ray-tracing
                return True
        return False

    def sense(self, observer, subject, initial=False):

        # can't sense, occluded
        if not initial and self._occluded(observer, subject):
            self._update_belief(observer, observer, subject) # just forward-integrate
            return None

        pos = self.sim.x['pH'][subject].copy()
        vel = self.sim.x['vH'][subject].copy()
        diff = pos - self.sim.x['pR'][observer]
        dist2 = diff.T @ diff

        # add noise
        alpha = (0.1, 0.01) # TODO: tune params
        pos[0] += random.gauss(0, alpha[0]*dist2 + alpha[1]*pos[0])
        pos[1] += random.gauss(0, alpha[0]*dist2 + alpha[1]*pos[1])
        vel[0] += random.gauss(0, alpha[0]*dist2 + alpha[1]*vel[0])
        vel[1] += random.gauss(0, alpha[0]*dist2 + alpha[1]*vel[1])

        self.obs[observer][subject]['pos'].append(pos)
        self.obs[observer][subject]['vel'].append(vel)

        if not initial:
            self._update_belief(observer, observer, subject)
        
        return (pos, vel)
    
    def share(self, sender, receiver, subject):
        self._update_belief(sender, receiver, subject)

    def test_share(self, sender, receiver, subject):
        return self._update_belief(sender, receiver, subject, update=False)