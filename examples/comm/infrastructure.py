import random

class Infrastructure:

    def __init__(self, sim):
        self.sim = sim
        self.obs = [[{'pos': [], 'vel': []} for _ in range(self.sim.m)] 
                    for _ in range(self.sim.n)]
        self.bel = [[{'pos': (0, 0), 'vel': (0, 0), 't': 0} for _ in range(self.sim.m)] 
                    for _ in range(self.sim.n)]
        self._init_bel()

    def _init_bel(self):
        s0 = (10, 10) # TODO: tune params
        for i in range(self.sim.n):
            for j in range(self.sim.m):
                p, v = self.sense(i, j, initial=True)
                self.bel[i][j]['pos'] = (p, s0[0]**2)
                self.bel[i][j]['vel'] = (v, s0[1]**2)

    # from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    # bayesian inference for gaussian prior and likelihood
    def _bayes_gauss(self, prior, lhood, obs):
        sigma2 = 1 / (len(obs) / lhood[1] + 1 / prior[1]) # eq. (20)
        # mu = sigma2 * (prior[0] / prior[1] + sum(obs) / lhood[1]) # eq. (24)
        mu = sigma2 * (prior[0] / prior[1] + (len(obs) * obs[-1]) / lhood[1]) # eq. (24) [MODIFIED TO REMOVE AVG]
        return (mu, sigma2)

    def _update_belief(self, sender, receiver, subject, update=True):

        # estimate current state from belief from last timestep
        vel_est = self.bel[receiver][subject]['vel'][0] # constant velocity assumption
        delta_t = self.sim.t - self.bel[receiver][subject]['t'] # time since last update
        pos_est = self.bel[receiver][subject]['pos'][0] + vel_est * delta_t # integrate to guess

        # prior is estimated state of receiver
        rec_dist = abs(self.sim.x['pR'][receiver] - self.bel[receiver][subject]['pos'][0])
        sigma0_v = 0.05 * rec_dist # TODO: tune param
        n = len(self.obs[receiver][subject]['pos'])
        # TODO: not sure if using dt below is correct, but using delta_t breaks it when delta_t=0
        prior_p = (pos_est, sigma0_v**2 * n * self.sim.dt) # (mu, sigma^2)
        prior_v = (vel_est, sigma0_v**2)
        
        # likelihood is latest observation of sender
        sen_dist = abs(self.sim.x['pR'][sender] - self.bel[sender][subject]['pos'][0])
        sigmaN = (0.25 * sen_dist, 0.25 * sen_dist) # TODO: tune params
        lhood_p = (self.obs[sender][subject]['pos'][-1], sigmaN[0]**2) # (mu, sigma^2)
        lhood_v = (self.obs[sender][subject]['vel'][-1], sigmaN[1]**2) 

        # compute posterior for receiver
        post_p = self._bayes_gauss(prior_p, lhood_p, self.obs[sender][subject]['pos']) # (mu, sigma^2)
        post_v = self._bayes_gauss(prior_v, lhood_v, self.obs[sender][subject]['vel'])

        print(sender, "->", receiver, "POS BAYESIAN", prior_p, lhood_p, post_p)
        print(sender, "->", receiver, "VEL BAYESIAN", prior_v, lhood_v, post_v)

        if update:
            self.bel[receiver][subject]['pos'] = post_p
            self.bel[receiver][subject]['vel'] = post_v
            self.bel[receiver][subject]['t'] = self.sim.t

        return (post_p, post_v)
    
    def sense(self, observer, subject, initial=False):
        pos = self.sim.x['pH'][subject]
        vel = self.sim.x['vH'][subject]
        dist = abs(pos - self.sim.x['pR'][observer])

        alpha = (0.01, 0.001) # TODO: tune params
        pos += random.gauss(0, alpha[0]*dist**2 + alpha[1]*pos)
        vel += random.gauss(0, alpha[0]*dist**2 + alpha[1]*vel)

        self.obs[observer][subject]['pos'].append(pos)
        self.obs[observer][subject]['vel'].append(vel)

        if not initial:
            self._update_belief(observer, observer, subject)
        
        return (pos, vel)
    
    def share(self, sender, receiver, subject):
        self._update_belief(sender, receiver, subject)

    def test_share(self, sender, receiver, subject):
        return self._update_belief(sender, receiver, subject, update=False)