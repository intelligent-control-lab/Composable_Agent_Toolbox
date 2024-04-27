from highway_simulator import HighwaySimulator
from infrastructure import Infrastructure
from knapsack import Knapsack
import ortoolpy
import random
from IDM import IDM

m = 1
n = 2
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

if __name__ == '__main__':

    x['pH'].append(3*L)
    x['vH'].append(20)
    x['aH'].append(0)
    x['lH'].append(-1)
    x['dH'].append(0)

    x['pR'].append(-3*L)
    x['vR'].append(20)
    x['aR'].append(0)
    x['lR'].append(-1)

    x['pR'].append(0)
    x['vR'].append(20)
    x['aR'].append(0)
    x['lR'].append(0)

    idmH = [IDM(random.uniform(0.5, 1.5), random.uniform(20, 30), 
            T, a, b, L) for _ in range(m)]
    idmR = [IDM(random.uniform(0.5, 1.5), random.uniform(20, 30), 
            T, a, b, L) for _ in range(n)]

    t = 0
    t_max = 40
    dt = 0.1

    sim = HighwaySimulator(x, m, n, q, L, idmH, idmR, dt)
    infra = Infrastructure(sim)
    knap = Knapsack(sim, infra, C, beta)
    
    while t <= t_max:
        
        # run individual sensing
        for r in range(n):
            for h in range(m):
                infra.sense(r, h)

        # run knapsack to decide communications
        comms = [(a, b, h) for a in range(n) for b in range(n) for h in range(m)] # consider all potential comms
        chosen = knap.sack(comms)[1] # indices of chosen comms

        # execute communications
        for i in chosen:
            (a, b, h) = comms[i]
            infra.share(a, b, h)

        # iterate simulation
        x = sim.move(use_idm=True)
        sim.vis(t, x['pH'][1])
        t += dt

# let's assume that 1. everyone wants information about everyone and that 2. everyone gets information about everyone.
# does everyone observe everyone??
# then, costs are just the full adjacancy matrix (C) and values are weighted sum of 1/d^2 and posterior - prior belief.

# perhaps everyone gets information about everyone else in their lane
# can observations be made across lanes? sure, since we can probably reasonably just consider longitudinal distance
# maybe if an agent wants to change lanes, it wants observations of humans in adjacent lane

# during car-following, agents get observations of car(s?) ahead in lane
# during lane-changing, agents get observations of car(s) in adjecent lane