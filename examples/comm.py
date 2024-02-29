import ortoolpy
import random
import copy
import math
import random
from IDM import IDM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

m = 2
n = 2
q = 0
beta = 100

C = [[random.randint(1, 10) for j in range(n)] 
        for i in range(n)]

v = [21, 11, 15, 9, 34, 25, 41, 52]
c = [22, 12, 16, 10, 35, 26, 42, 53]
print(ortoolpy.knapsack(v, c, beta))



dt = 0.1
t_max = 40

L = 5

x = {'pH': [], 'vH': [], 'aH': [], 'lH': [], 'dH': [], 
     'pR': [], 'vR': [], 'aR': [], 'lR': [], 
     'pB': [], 'vB': [], 'lB': []}

a_max = 2.0
b_max = 4.0
vR_max = [35, 35]

s0 = 0.3*L
v0 = 35
vH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1

idmH = [IDM(random.uniform(0.5, 1.5), random.uniform(20, 30), 
            T, a, b, L) for _ in range(m)]
idmR = [IDM(random.uniform(0.5, 1.5), random.uniform(20, 30), 
            T, a, b, L) for _ in range(n)]

x_all = []

t = 0

fig, ax = plt.subplots()

def vis(pov):
    
    ax.cla()
    ax.axis([-4*L, 4*L, pov - 4*L, pov + 4*L])
    ax.axvline(x=-L/2, color="black", linestyle="--")
    ax.axvline(x=L/2, color="black", linestyle="--")
    ax.axvline(x=-3*L/2, color="black", linestyle="-")
    ax.axvline(x=3*L/2, color="black", linestyle="-")

    ax.text(-4*L, pov + 4*L, "t = " + str(round(t, 1)))
    for i in range(m):
        ax.text(x['lH'][i]*L, x['pH'][i], str(round(x['vH'][i], 1)))
    for i in range(n):
        ax.text(x['lR'][i]*L, x['pR'][i], str(round(x['vR'][i], 1)))
    for i in range(q):
        ax.text(x['lB'][i]*L, x['pB'][i], str(round(x['vB'][i], 1)))

    for i in range(m):
        ax.add_patch(Rectangle(
            (x['lH'][i]*L - L/4, x['pH'][i] - L/2), 
            L/2, L, 
            facecolor='blue'))
    for i in range(n):
        ax.add_patch(Rectangle(
            (x['lR'][i]*L - L/4, x['pR'][i] - L/2), 
            L/2, L, 
            facecolor='red'))
    for i in range(q):
        ax.add_patch(Rectangle(
            (x['lB'][i]*L - L/4, x['pB'][i] - L/2), 
            L/2, L, 
            facecolor='green'))

    plt.pause(0.001)

def get_following(i, l):
    fol = -1
    for j in range(m):
        if j == i or x['lH'][j] != l or x['pH'][i] > x['pH'][j]:
            continue
        if fol == -1 or x['pH'][j] < x['pH'][fol]:
            fol = j
    if fol != -1:
        return x['pH'][fol], x['vH'][fol]
    for j in range(n):
        if x['lR'][j] != l or x['pH'][i] > x['pR'][j]:
            continue
        if fol == -1 or x['pR'][j] < x['pR'][fol]:
            fol = j
    if fol != -1:
        return x['pR'][fol], x['vR'][fol]
    for j in range(q):
        if x['lB'][j] != l or x['pH'][i] > x['pB'][j]:
            continue
        if fol == -1 or x['pB'][j] < x['pB'][fol]:
            fol = j
    if fol != -1:
        return x['pB'][fol], x['vB'][fol]
    return None

def is_occupied(p, l):
    for i in range(m):
        if x['lH'][i] == l and abs(p - x['pH'][i]) < L + L:
            return True
    for j in range(n):
        if x['lR'][j] == l and abs(p - x['pR'][j]) < L + L:
            return True
    for k in range(q):
        if x['lB'][k] == l and abs(p - x['pB'][k]) < L + L:
            return True
    return False

def f():

    aH = []
    dH = []

    for i in range(m):
        fol = get_following(i, x['lH'][i])
        aH.append(idmH[i].free_road(x['vH'][i]) if fol is None 
                  else idmH[i].idm(x['pH'][i], fol[0], x['vH'][i], fol[1]))
        incentive = False
        if fol is not None:
            fol_new = get_following(i, -1)
            vel_new = v0 if fol_new is None else fol_new[1]
            incentive = vel_new - fol[1] >= vH_th
        safety = not is_occupied(x['pH'][i], x['lH'][i] - 1)
        # dH.append(0 if fol is None 
        #           else -(incentive and safety))
        dH.append(0 if fol is None else -safety)
        
    return (aH, dH)

def u():

    aR = []
    dR = [0 for _ in range(n)]

    for i in range(n):
        fol = get_following(i, x['lR'][i])
        aR.append(idmR[i].free_road(x['vR'][i]) if fol is None 
                  else idmR[i].idm(x['pR'][i], fol[0], x['vR'][i], fol[1]))
    
    return (aR, dR)

def move():

    aH, dH = f()
    aR, dR = u()

    for i in range(m):
        aH[i] = min(a_max, max(-b_max, aH[i]))
        x['vH'][i] += aH[i] * dt
        x['pH'][i] += x['vH'][i] * dt
        x['lH'][i] = max(-1, min(1, 
            x['lH'][i] + round(dH[i])))
        
    for i in range(n):
        aR[i] = min(a_max, max(-b_max, aR[i]))
        x['vR'][i] += aR[i] * dt
        x['vR'][i] = min(vR_max[i], x['vR'][i])
        x['pR'][i] += x['vR'][i] * dt
        x['lR'][i] = max(-1, min(1, 
            x['lR'][i] + round(dR[i])))
        
    for i in range(q):
        x['pB'][i] += x['vB'][i] * dt
        
    x['aH'] = aH
    x['aR'] = aR
    x_all.append((t, copy.deepcopy(x)))

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

    # u = ([0 for _ in range(n)], [0 for _ in range(n)])
    while t <= t_max:
        move()
        vis(x['pH'][1])
        t += dt
