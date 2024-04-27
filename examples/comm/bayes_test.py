from highway_simulator import HighwaySimulator
from infrastructure import Infrastructure
from knapsack import Knapsack
import random
from IDM import IDM

import matplotlib.pyplot as plt

t = 0
t_max = 100000
dt = 0.1

m = 1
n = 1
q = 0
beta = 100

C = [[random.randint(1, 10) for j in range(n)] 
        for i in range(n)]

L = 5
a_max = 2.0
b_max = 4.0
s0 = 0.3*L
v0 = 35
vH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1

x = {'pH': [], 'vH': [], 'aH': [], 'lH': [], 'dH': [], 
     'pR': [], 'vR': [], 'aR': [], 'lR': [], 
     'pB': [], 'vB': [], 'lB': []}

import numpy as np
import scipy.stats as stats
import math

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

def run_sense(infra):
    p, v = infra.sense(0, 0)
    print(f'MEASURED {p} {v}')

    # if t % 5 != 0:
    #     return

    ax1.cla()
    ax2.cla()

    mu_p, sigma2_p = infra.bel[0][0]['pos']
    sigma_p = np.sqrt(sigma2_p)
    mu_v, sigma2_v = infra.bel[0][0]['vel']
    sigma_v = np.sqrt(sigma2_v)

    x_p = np.linspace(mu_p - 3*sigma_p, mu_p + 3*sigma_p, 100)
    x_v = np.linspace(mu_v - 3*sigma_v, mu_v + 3*sigma_v, 100)

    ax1.plot(x_p, stats.norm.pdf(x_p, mu_p, sigma_p), c='b')
    ax2.plot(x_v, stats.norm.pdf(x_v, mu_v, sigma_v), c='r')

    # ax1.plot(infra.sim.x['pH'][0], 1)
    # ax2.plot(infra.sim.x['vH'][0], 1)

    # plt.show()
    plt.pause(0.01)

if __name__ == '__main__':

    x['pH'].append(30)
    x['vH'].append(20)
    x['aH'].append(0)
    x['lH'].append(0)
    x['dH'].append(0)

    x['pR'].append(0)
    x['vR'].append(20)
    x['aR'].append(0)
    x['lR'].append(0)

    idmH = [IDM(random.uniform(0.5, 1.5), random.uniform(20, 30), 
            T, a, b, L) for _ in range(m)]
    idmR = [IDM(random.uniform(0.5, 1.5), random.uniform(20, 30), 
            T, a, b, L) for _ in range(n)]

    sim = HighwaySimulator(x, m, n, q, L, idmH, idmR, dt)
    infra = Infrastructure(sim)
    knap = Knapsack(sim, infra, C, beta)
    
    while t <= t_max:
        print(t)
        x = sim.move(use_idm=True)
        # sim.vis(t, x['pR'][0])
        run_sense(infra)
        t += dt
        input()
