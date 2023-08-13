import copy
import math
import random
from IDM import IDM
from CBF_hF import CBF_hF
from CBF_hV import CBF_hV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxopt

m = 1
n = 1
q = 0

dt = 0.1
t_max = 60

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
v0 = 36
vH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1
idm = IDM(s0, v0, T, a, b, L)

dist_min = 1.5 * L
const_vel = [False, True]

x_all = []

t = 0
safe = False

hF = None
hV = None

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
    ax.add_patch(Rectangle(
            (x['lR'][i]*L - L/4, x['pR'][i] - L/2 - s_min), 
            L/2, s_min,
            facecolor=(1,1,0,0.5)))

    if x['lH'][0] != -1:
        ax.add_patch(Rectangle(
                (-1*L - L/4, x['pH'][0] - dist_min / 2), 
                L/2, dist_min,
                facecolor=(1,1,0,0.5)))

    plt.pause(0.001)

def plot():

    global x_all
    x_all = x_all[10:]

    xs = [s[0] for s in x_all]

    fig0, ax0 = plt.subplots()
    y_pH = [s[1][0] for s in x_all]
    y_pR = [s[2][0] for s in x_all]
    y = [r - h for h, r in zip(y_pH, y_pR)]
    ax0.plot(xs, y, label='dist')
    # plt.axhline(y=s_min+L, color='black', linestyle='--')
    # ax0.plot(xs, y_pH, label='pH', c='b')
    # ax0.plot(xs, y_pR, label='pR', c='r')
    ax0.legend(loc='upper right')

    fig1, ax1 = plt.subplots()
    y = [s[3][0] for s in x_all]
    ax1.plot(xs, y, label='vH')
    plt.axhline(y=v_max, color='black', linestyle='--')
    ax1.legend(loc='upper right')

    fig2, ax2 = plt.subplots()
    y_vH = [s[3][0] for s in x_all]
    y_vR = [s[4][0] for s in x_all]
    ax2.plot(xs, y_vH, label='vH', c='b')
    ax2.plot(xs, y_vR, label='vR', c='r')
    ax2.legend(loc='upper right')

    fig3, ax3 = plt.subplots()
    y_u = [s[6][0] for s in x_all]
    ax3.plot(xs, y_u, label='u')
    ax3.legend(loc='lower right')

    fig4, ax4 = plt.subplots()
    y_aH = [s[5][0] for s in x_all]
    y_aR = [s[6][0] for s in x_all]
    ax4.plot(xs, y_aH, label='aH', c='b')
    ax4.plot(xs, y_aR, label='aR', c='r')
    ax4.legend(loc='lower right')

    plt.show()

def safe(): 
    return x['pR'][0] - x['pH'][0] - L >= s_min or x['lH'][0] != x['lR'][0]

def get_following(i):
    f = -1
    for j in range(m):
        if i == j:
            continue
        if x['lH'][i] != x['lH'][j] or x['pH'][i] > x['pH'][j]:
            continue
        if f == -1 or x['pH'][j] < x['pH'][f]:
            f = j
    if f != -1:
        return x['pH'][f], x['vH'][f]
    for j in range(n):
        if x['lH'][i] != x['lR'][j] or x['pH'][i] > x['pR'][j]:
            continue
        if f == -1 or x['pR'][j] < x['pR'][f]:
            f = j
    if f != -1:
        return x['pR'][f], x['vR'][f]
    return None

def get_new_following(i, l):
    f = -1
    for j in range(m):
        if x['lH'][j] != l or x['pH'][i] > x['pH'][j]:
            continue
        if f == -1 or x['pH'][j] < x['pH'][f]:
            f = j
    if f != -1:
        return x['pH'][f], x['vH'][f]
    for j in range(n):
        if x['lR'][j] != l or x['pH'][i] > x['pR'][j]:
            continue
        if f == -1 or x['pR'][j] < x['pR'][f]:
            f = j
    if f != -1:
        return x['pR'][f], x['vR'][f]
    return None

def is_occupied(p, l):
    for i in range(m):
        if x['lH'][i] == l and abs(p - x['pH'][i]) < dist_min:
            return True
    for i in range(n):
        if x['lR'][i] == l and abs(p - x['pR'][i]) < dist_min:
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
        
    x['aH'] = aH
    x['aR'] = aR
    x_all.append((t, copy.deepcopy(x)))

def f():

    aH = []
    dH = []

    for i in range(m):
        if const_vel[i]:
            fol = get_following(i)
            if fol == None:
                aH.append(0)
            else:
                aH.append(idm.idm(x['pH'][i], fol[0], x['vH'][i], fol[1]))
            dH.append(0)
            continue
        
        f = get_following(i)
        aH.append(idm.free_road(x['vH'][i]) if f is None 
                  else idm.idm(x['pH'][i], f[0], x['vH'][i], f[1]))
        
        incentive = False
        if f is not None:
            f_new = get_new_following(i, -1)
            vel_new = v0 if f_new is None else f_new[1]
            incentive = vel_new - f[1] >= vH_th

        safety = not is_occupied(x['pH'][i], x['lH'][i] - 1)
        # dH.append(0 if f is None 
        #           else -(incentive and safety))
        dH.append(0)
        
    return (aH, dH)

def compute_u():

    # Following distance lower bound
    lb = hF.lb(x)

    # Velocity upper bound
    ub = hV.ub(x)

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
    u_star = max(0, lb)
    # u_star = min(0, ub)

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

    hF = CBF_hF(x, s_min, L, dt, c_min, idm)
    hV = CBF_hV(x, v_max, c_min, idm)

    u = ([0 for _ in range(n)], [0 for _ in range(n)])
    while t <= t_max:
        apply_control(compute_u())
        vis(min(x['pH'][0] + s_min, x['pR'][0]))
        t += dt

    plot()
