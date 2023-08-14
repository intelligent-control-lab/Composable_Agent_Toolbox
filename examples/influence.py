import copy
import math
import random
from IDM import IDM
from CBF_hF import CBF_hF
from CBF_hV import CBF_hV
from CBF_hL import CBF_hL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxopt

m = 1
n = 1
q = 1

dt = 0.1
t_max = 15

t_switch = -1

L = 5

x = {'pH': [], 'vH': [], 'aH': [], 'lH': [], 'dH': [], 
     'pR': [], 'vR': [], 'aR': [], 'lR': [], 
     'pB': [], 'vB': [], 'lB': []}

c_min = 1

s_min = 10.0001 - L
a_max = 2.0
b_max = 4.0
vR_max = 40
v_max = 25

s0 = 0.3*L
v0 = 30
vH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1
idm = IDM(s0, v0, T, a, b, L)

dist_min = 10.0001 - L

x_all = []

t = 0
safe = False

hF = None
hV = None
hL = None

fig, ax = plt.subplots()

def vis(pov):
    
    ax.cla()
    ax.axis([-4*L, 4*L, pov - 4*L, pov + 4*L])
    ax.axvline(x=-L/2, color="black", linestyle="--")
    ax.axvline(x=L/2, color="black", linestyle="--")
    ax.axvline(x=-3*L/2, color="black", linestyle="-")
    ax.axvline(x=3*L/2, color="black", linestyle="-")

    ax.text(-4*L, pov - 4*L, "Safe: " + str(safe()))
    ax.text(-4*L, pov + 4*L, "t = " + str(t))
    for i in range(m):
        ax.text(x['lH'][i]*L, x['pH'][i], str(x['vH'][i]))
    for i in range(n):
        ax.text(x['lR'][i]*L, x['pR'][i], str(x['vR'][i]))
    for i in range(q):
        ax.text(x['lB'][i]*L, x['pB'][i], str(x['vB'][i]))

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

    ax.add_patch(Rectangle(
            (x['lR'][i]*L - L/4, x['pR'][i] - L/2 - s_min), 
            L/2, s_min,
            facecolor=(1,1,0,0.5)))

    if x['lH'][0] != -1:
        # ax.add_patch(Rectangle(
        #         (-1*L - L/4, x['pH'][0] - dist_min / 2), 
        #         L/2, dist_min,
        #         facecolor=(1,1,0,0.5)))
        ax.add_patch(Rectangle(
            (x['lB'][i]*L - L/4, x['pB'][i] - L/2 - dist_min), 
            L/2, dist_min,
            facecolor=(1,1,0,0.5)))

    plt.pause(0.001)

def plot():

    global x_all
    x_all = x_all[10:]

    xs = [s[0] for s in x_all]

    # fig0, ax0 = plt.subplots()
    # y_pH = [s[1]['pH'][0] for s in x_all]
    # y_pR = [s[1]['pR'][0] for s in x_all]
    # y = [r - h for h, r in zip(y_pH, y_pR)]
    # ax0.plot(xs, y, label='dist')
    # # plt.axhline(y=s_min+L, color='black', linestyle='--')
    # # ax0.plot(xs, y_pH, label='pH', c='b')
    # # ax0.plot(xs, y_pR, label='pR', c='r')
    # ax0.legend(loc='upper right')

    fig0, ax0 = plt.subplots()
    y_pH = [s[1]['pH'][0] for s in x_all]
    y_pB = [s[1]['pB'][0] for s in x_all]
    y = [r - h for h, r in zip(y_pH, y_pB)]
    ax0.plot(xs, y, label='pB - pH')
    plt.axhline(y=dist_min+L, color='black', linestyle='--')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax0.legend(loc='lower right')

    # fig1, ax1 = plt.subplots()
    # y = [s[1]['vH'][0] for s in x_all]
    # ax1.plot(xs, y, label='vH')
    # plt.axhline(y=v_max, color='black', linestyle='--')
    # ax1.legend(loc='upper right')

    fig2, ax2 = plt.subplots()
    y_vH = [s[1]['vH'][0] for s in x_all]
    y_vR = [s[1]['vR'][0] for s in x_all]
    y_vB = [s[1]['vB'][0] for s in x_all]
    ax2.plot(xs, y_vH, label='vH', c='b')
    ax2.plot(xs, y_vR, label='vR', c='r')
    ax2.plot(xs, y_vB, label='vB', c='g')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax2.legend(loc='lower right')

    fig3, ax3 = plt.subplots()
    y_u = [s[1]['aH'][0] for s in x_all]
    ax3.plot(xs, y_u, label='u')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax3.legend(loc='upper right')

    fig4, ax4 = plt.subplots()
    y_aH = [s[1]['aH'][0] for s in x_all]
    y_aR = [s[1]['aR'][0] for s in x_all]
    ax4.plot(xs, y_aH, label='aH', c='b')
    ax4.plot(xs, y_aR, label='aR', c='r')
    plt.axvline(x=t_switch, color='gray', linestyle='--')
    ax4.legend(loc='upper right')

    plt.show()

def safe(): 
    return x['pR'][0] - x['pH'][0] - L >= s_min or x['lH'][0] != x['lR'][0]

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
        if x['lH'][i] == l and abs(p - x['pH'][i]) < dist_min + L:
            return True
    for j in range(n):
        if x['lR'][j] == l and abs(p - x['pR'][j]) < dist_min + L:
            return True
    for k in range(q):
        if x['lB'][k] == l and abs(p - x['pB'][k]) < dist_min + L:
            return True
    return False

def apply_control(u):

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
        x['vR'][i] = min(vR_max, x['vR'][i])
        x['pR'][i] += x['vR'][i] * dt
        x['lR'][i] = max(-1, min(1, 
            x['lR'][i] + round(dR[i])))
        
    for i in range(q):
        x['pB'][i] += x['vB'][i] * dt
        
    x['aH'] = aH
    x['aR'] = aR
    x_all.append((t, copy.deepcopy(x)))

def f():

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
        # dH.append(0 if fol is None 
        #           else -(incentive and safety))
        global t_switch
        if safety and t_switch == -1:
            t_switch = t
        dH.append(0 if fol is None else -safety)
        # dH.append(0)
        
    return (aH, dH)

def compute_u():

    # # Following distance lower bound
    # lb = hF.lb(x)

    # # Velocity upper bound
    # ub = hV.ub(x)

    # Lane change
    ub = hL.ub(x)

    # u_star = 0
    # if lb <= ub:
    #     sol = cvxopt.solvers.qp(cvxopt.matrix([[1.0]]), # 1.0 for min, -1.0 for max
    #                             cvxopt.matrix([0.0]), 
    #                             cvxopt.matrix([[-1.0, 1.0]]), 
    #                             cvxopt.matrix([-lb, ub]))
    #     u_star = sol['x'][0]
    #     print(sol)
    # else:
    #     print("INFEASIBLE")

    # J(u) = ||u||^2
    # u_star = max(0, lb)
    # u_star = 0
    u_star = min(0, ub)

    if x['lH'][0] != x['lR'][0]:
        u_star = idm.free_road(x['vR'][0])

    print(f"U_STAR: {u_star}")
    return ([u_star for _ in range(n)], 
            [0 for _ in range(n)])

if __name__ == '__main__':

    x['pH'].append(-2*L)
    x['vH'].append(30)
    x['aH'].append(0)
    x['lH'].append(0)
    x['dH'].append(0)

    x['pR'].append(0)
    x['vR'].append(26)
    x['aR'].append(0)
    x['lR'].append(0)

    x['pB'].append(-2*L)
    x['vB'].append(25)
    x['lB'].append(-1)

    hF = CBF_hF(x, s_min, L, dt, c_min, idm)
    hV = CBF_hV(x, v_max, c_min, idm)
    hL = CBF_hL(x, s_min, L, c_min, idm)

    u = ([0 for _ in range(n)], [0 for _ in range(n)])
    while t <= t_max:
        apply_control(compute_u())
        vis(min(x['pH'][0] + s_min, x['pR'][0]))
        t += dt

    plot()
