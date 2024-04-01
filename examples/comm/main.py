from highway_simulator import HighwaySimulator
from infrastructure import Infrastructure
import ortoolpy
import random
from IDM import IDM

m = 2
n = 2
q = 0
beta = 100

C = [[random.randint(1, 10) for j in range(n)] 
        for i in range(n)]

v = [21, 11, 15, 9, 34, 25, 41, 52]
c = [22, 12, 16, 10, 35, 26, 42, 53]
print(ortoolpy.knapsack(v, c, beta))

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

    x['pH'].append(-3*L)
    x['vH'].append(20)
    x['aH'].append(0)
    x['lH'].append(-1)
    x['dH'].append(0)

    x['pH'].append(-2*L)
    x['vH'].append(20)
    x['aH'].append(0)
    x['lH'].append(0)
    x['dH'].append(0)

    x['pR'].append(0)
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
    
    while t <= t_max:
        x = sim.move(use_idm=True)
        sim.vis(t, x['pH'][1])
        t += dt

# let's assume that 1. everyone wants information about everyone and that 2. everyone gets information about everyone.
# does everyone observe everyone??
# then, costs are just the full adjacancy matrix (C) and values are weighted sum of 1/d^2 and posterior - prior belief.
