from highway_simulator import HighwaySimulator
from infrastructure import Infrastructure
from knapsack import Knapsack
import ortoolpy
import random
from IDM import IDM

m = 1
n = 2
q = 0

# C = [[random.randint(1, 10) for j in range(n)] 
#         for i in range(n)]

C = [[1 for j in range(n)] for i in range(n)]
beta = 1

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

    t_max = 5
    dt = 0.1

    sim = HighwaySimulator(x, m, n, q, L, idmH, idmR, dt)
    infra = Infrastructure(sim)
    knap = Knapsack(sim, infra, C, beta)
    
    while sim.t <= t_max:

        # iterate simulation
        x = sim.move(use_idm=False)
        sim.vis(x['pR'][1])

        # run individual sensing
        for r in range(n):
            for h in range(m):
                infra.sense(r, h)

        # run knapsack to decide communications
        comms = [(a, b, h) for a in range(n) for b in range(n) if a != b for h in range(m)] # consider all potential comms
        print(comms)
        chosen = knap.sack(comms)[1] # indices of chosen comms
        print(chosen)

        # execute communications
        for i in chosen:
            (a, b, h) = comms[i]
            infra.share(a, b, h)
