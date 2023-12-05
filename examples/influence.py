import copy
import math
import random
from IDM import IDM
from CBF_hF import CBF_hF
from CBF_hV import CBF_hV
from CBF_hL import CBF_hL
from CBF_hF31 import CBF_hF31
from CBF_hB31 import CBF_hB31
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxopt

m = 2
n = 2
q = 0

dt = 0.01
t_max = 30

t_switch = -1

L = 5

x = {'pH': [], 'vH': [], 'aH': [], 'lH': [], 'dH': [], 
     'pR': [], 'vR': [], 'aR': [], 'lR': [], 
     'pB': [], 'vB': [], 'lB': []}

c_min = 3.25

s_min = 10.0001 - L
a_max = 2.0
b_max = 4.0
vR_max = [35, 35]

s0 = 0.3*L
v0 = 35
vH_th = 10
T = 0.01
a = 1.8
b = 3.1
idm = IDM(s0, v0, T, a, b, L)

dist_min = 10.0001 - L

x_all = []

t = 0
safe = False

hF = None
hB = None

fig, ax = plt.subplots()

def vis(pov): # simulation visualizer
    
    ax.cla()
    ax.axis([-4*L, 4*L, pov - 4*L, pov + 4*L])
    ax.axvline(x=-L/2, color="black", linestyle="--")
    ax.axvline(x=L/2, color="black", linestyle="--")
    ax.axvline(x=-3*L/2, color="black", linestyle="-")
    ax.axvline(x=3*L/2, color="black", linestyle="-")

    ax.text(-4*L, pov + 4*L, "t = " + str(round(t, 2)))
    for i in range(m):
        ax.text(x['lH'][i]*L - L/4, x['pH'][i], str(round(x['vH'][i], 1)))
    for i in range(n):
        ax.text(x['lR'][i]*L - L/4, x['pR'][i], str(round(x['vR'][i], 1)))
    for i in range(q):
        ax.text(x['lB'][i]*L - L/4, x['pB'][i], str(round(x['vB'][i], 1)))

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

    if round(t, 2) % 2.5 == 0:
        plt.savefig(f'cbf_31_vis-t{round(t, 2)}.png')

def plot(): # pos, vel, acc vs time plots

    global x_all

    xs = [s[0] for s in x_all]

    fig0, ax0 = plt.subplots()
    y_pR = [s[1]['pR'][0] for s in x_all]
    y_pH = [s[1]['pH'][0] for s in x_all]
    y = [r - h for r, h in zip(y_pR, y_pH)]
    ax0.plot(xs, y, label='pR1 - pH1')
    plt.axhline(y=30, color='black', linestyle='--')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax0.legend(loc='lower right')

    fig1, ax1 = plt.subplots()
    y_pR1 = [s[1]['pR'][0] - s[1]['pH'][1] for s in x_all]
    y_pH1 = [s[1]['pH'][0] - s[1]['pH'][1] for s in x_all]
    y_pR2 = [s[1]['pR'][1] - s[1]['pH'][1] for s in x_all]
    ax1.plot(xs, y_pR1, label='pR1 - pH2')
    ax1.plot(xs, y_pH1, label='pH1 - pH2')
    ax1.plot(xs, y_pR2, label='pR2 - pH2')
    plt.axhline(y=10, color='black', linestyle='--')
    plt.axhline(y=-10, color='black', linestyle='--')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax1.legend(loc='lower left')

    fig111, ax111 = plt.subplots()
    y_pR1 = [s[1]['pR'][0] - s[1]['pH'][0] for s in x_all]
    y_pH2 = [s[1]['pH'][1] - s[1]['pH'][0] for s in x_all]
    y_pR2 = [s[1]['pR'][1] - s[1]['pH'][0] for s in x_all]
    ax111.plot(xs, y_pR1, label='pR1 - pH1')
    ax111.plot(xs, y_pH2, label='pH2 - pH1')
    ax111.plot(xs, y_pR2, label='pR2 - pH1')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    plt.axhline(y=10, color='black', linestyle='--')
    ax111.legend(loc='upper left')

    fig2, ax2 = plt.subplots()
    y_vR1 = [s[1]['vR'][0] for s in x_all]
    y_vH1 = [s[1]['vH'][0] for s in x_all]
    y_vR2 = [s[1]['vR'][1] for s in x_all]
    y_vH2 = [s[1]['vH'][1] for s in x_all]
    ax2.plot(xs, y_vR1, label='vR1')
    ax2.plot(xs, y_vH1, label='vH1')
    ax2.plot(xs, y_vR2, label='vR2')
    ax2.plot(xs, y_vH2, label='vH2')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax2.legend(loc='lower right')

    fig3, ax3 = plt.subplots()
    y_aR1 = [s[1]['aR'][0] for s in x_all]
    y_aH1 = [s[1]['aH'][0] for s in x_all]
    y_aR2 = [s[1]['aR'][1] for s in x_all]
    y_aH2 = [s[1]['aH'][1] for s in x_all]
    ax3.plot(xs, y_aH1, label='aH1')
    ax3.plot(xs, y_aH2, label='aH2')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax3.legend(loc='lower left')

    fig4, ax4 = plt.subplots()
    y_u1 = [s[1]['aR'][0] for s in x_all]
    y_u2 = [s[1]['aR'][1] for s in x_all]
    ax4.plot(xs, y_u1, label='u1')
    ax4.plot(xs, y_u2, label='u2')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax4.legend(loc='upper right')

    fig5, ax5 = plt.subplots()
    y_vR1 = [s[1]['vR'][0] for s in x_all]
    y_vR2 = [s[1]['vR'][1] for s in x_all]
    y_dv = [r1 - r2 + 0.2 for r1, r2 in zip(y_vR1, y_vR2)]
    ax5.plot(xs, y_dv, label='vR1 - vR2')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    plt.axhline(y=3.97, color='black', linestyle='--')
    ax5.legend(loc='lower right')
     
    plt.show()

def safe(): # whether switching lanes is safe
    return x['pR'][0] - x['pH'][0] - L >= s_min or x['lH'][0] != x['lR'][0]

def get_following(i, l): # pos, vel of car followed by car i in lane l
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

def is_occupied(p, l): # whether pos p in lane l is occupied
    for i in range(m):
        if x['lH'][i] == l and abs(p - x['pH'][i]) < dist_min + L:
            return True
    for j in range(n):
        if x['lR'][j] == l and abs(p - x['pR'][j]) < dist_min + L:
            return True
    for k in range(q):
        if x['lB'][k] == l and abs(p - x['pB'][k]) < dist_min + L:
            return True
    return False

def apply_control(u): # apply controls u to robot cars

    aH, dH = f()
    aR, dR = u

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

def f(): # compute human car controls

    aH = []
    dH = []

    for i in range(m):
        fol = get_following(i, x['lH'][i])
        aH.append(idm.free_road(x['vH'][i]) if fol is None 
                  else idm.idm(x['pH'][i], fol[0], x['vH'][i], fol[1]))
        incentive = False
        if fol is not None:
            fol_new = get_following(i, -1)
            vel_new = v0 if fol_new is None else fol_new[1]
            incentive = vel_new - fol[1] >= vH_th
        safety = not is_occupied(x['pH'][i], x['lH'][i] - 1)
        global t_switch
        if i == 1 and safety and t_switch == -1:
            t_switch = t
        dH.append(0 if fol is None else -safety)
        
    return (aH, dH)

def compute_u(): # compute robot car controls

    # get constraint parameters
    A11, A12, b1 = hF.constraint(x)
    A21, A22, b2 = hB.constraint(x)
    print(A11, A12, b1, A21, A22, b2)

    # Ax >= b
    A = cvxopt.matrix([[-A11, -A21], [-A12, -A22]])
    print(A)
    b = cvxopt.matrix([-b1, -b2])
    print(b)

    # minimize ||u||^2 (control effort) s.t. constraints
    u_star = [0 for _ in range(n)]
    sol = cvxopt.solvers.qp(cvxopt.matrix([[1.0, 0.0], [0.0, 1.0]]),
                            cvxopt.matrix([0.0, 0.0]), 
                            A, b)
    if sol['status'] == 'optimal':
        u_star = sol['x']

    # if objective already satisfied, passive driving
    if x['lH'][1] == -1:
        u_star[0] = idm.free_road(x['vR'][0])
        u_star[1] = idm.free_road(x['vR'][1])

    print(f"U_STAR: {u_star}")
    return (u_star, [0 for _ in range(n)])

if __name__ == '__main__':

    # set up initial conditions
    x['pH'].append(-2*L)
    x['vH'].append(30)
    x['aH'].append(0)
    x['lH'].append(-1)
    x['dH'].append(0)

    x['pH'].append(-2*L)
    x['vH'].append(30)
    x['aH'].append(0)
    x['lH'].append(0)
    x['dH'].append(0)

    x['pR'].append(0)
    x['vR'].append(30)
    x['aR'].append(0)
    x['lR'].append(-1)

    x['pR'].append(0)
    x['vR'].append(30)
    x['aR'].append(0)
    x['lR'].append(0)

    # initialize cbf functions
    hF = CBF_hF31(x, s_min, L, dt, c_min, idm)
    hB = CBF_hB31(x, s_min, L, dt, c_min, idm)

    # run simulation
    u = ([0 for _ in range(n)], [0 for _ in range(n)])
    while t <= t_max:
        apply_control(compute_u())
        vis((x['pH'][0] + x['pH'][1] + x['pR'][1] + x['pR'][0]) / 4)
        t += dt

    vis((x['pH'][0] + x['pH'][1] + x['pR'][1] + x['pR'][0]) / 4)

    plot()
