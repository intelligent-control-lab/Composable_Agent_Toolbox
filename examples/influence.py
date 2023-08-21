import copy
import math
import random
from IDM import IDM
from CBF_hF1_33 import CBF_hF1_33
from CBF_hF2_33 import CBF_hF2_33
from CBF_hB1_33 import CBF_hB1_33
from CBF_hB2_33 import CBF_hB2_33
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxopt

m = 2
n = 3
q = 0

dt = 0.1
t_max = 20

t_switch = [-1, -1]

L = 5

x = {'pH': [], 'vH': [], 'aH': [], 'lH': [], 'dH': [], 
     'pR': [], 'vR': [], 'aR': [], 'lR': [], 
     'pB': [], 'vB': [], 'lB': []}

# c_min = 0.005
# c_min = 1
c_min = 0.05

s_min = 10.0001 - L
a_max = 2.0
b_max = 4.0
# vR_max = [34.5, 38]
vR_max = [35, 35, 35]

s0 = 0.3*L
v0 = 35
dvH_th = 4.1
T = 0.01 # 1.3
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

    # ax.add_patch(Rectangle(
    #         (x['lR'][i]*L - L/4, x['pR'][i] - L/2 - s_min), 
    #         L/2, s_min,
    #         facecolor=(1,1,0,0.5)))

    # if x['lH'][0] != -1:
    #     # ax.add_patch(Rectangle(
    #     #         (-1*L - L/4, x['pH'][0] - dist_min / 2), 
    #     #         L/2, dist_min,
    #     #         facecolor=(1,1,0,0.5)))
    #     ax.add_patch(Rectangle(
    #         (x['lB'][i]*L - L/4, x['pB'][i] - L/2 - dist_min), 
    #         L/2, dist_min,
    #         facecolor=(1,1,0,0.5)))

    plt.pause(0.001)

def plot():

    # TODO: need to redo scenario 21 graph labels

    global x_all
    # x_all = x_all[10:]

    xs = [s[0] for s in x_all]

    fig1, ax1 = plt.subplots()
    y_pR1 = [s[1]['pR'][0] - s[1]['pH'][0] for s in x_all]
    y_pR2 = [s[1]['pR'][1] - s[1]['pH'][0] for s in x_all]
    y_pR3 = [s[1]['pR'][2] - s[1]['pH'][0] for s in x_all]
    y_pH2 = [s[1]['pH'][1] - s[1]['pH'][0] for s in x_all]
    ax1.plot(xs, y_pR1, label='pR1 - pH1')
    ax1.plot(xs, y_pR2, label='pR2 - pH1')
    ax1.plot(xs, y_pR3, label='pR3 - pH1')
    ax1.plot(xs, y_pH2, label='pH2 - pH1')
    plt.axhline(y=10, color='black', linestyle='--')
    plt.axhline(y=-10, color='black', linestyle='--')
    plt.axvline(x=t_switch[0], color='gray', linestyle='--')
    plt.axvline(x=t_switch[1], color='gray', linestyle='dotted')
    ax1.legend(loc='lower left')

    fig0, ax0 = plt.subplots()
    y_pR1 = [s[1]['pR'][0] - s[1]['pH'][1] for s in x_all]
    y_pR2 = [s[1]['pR'][1] - s[1]['pH'][1] for s in x_all]
    y_pR3 = [s[1]['pR'][2] - s[1]['pH'][1] for s in x_all]
    y_pH1 = [s[1]['pH'][0] - s[1]['pH'][1] for s in x_all]
    ax0.plot(xs, y_pR1, label='pR1 - pH2')
    ax0.plot(xs, y_pR2, label='pR2 - pH2')
    ax0.plot(xs, y_pR3, label='pR3 - pH2')
    ax0.plot(xs, y_pH1, label='pH1 - pH2')
    plt.axhline(y=10, color='black', linestyle='--')
    plt.axhline(y=-10, color='black', linestyle='--')
    plt.axvline(x=t_switch[0], color='gray', linestyle='--')
    plt.axvline(x=t_switch[1], color='gray', linestyle='dotted')
    ax0.legend(loc='lower left')

    fig2, ax2 = plt.subplots()
    y_vR1 = [s[1]['vR'][0] for s in x_all]
    y_vR2 = [s[1]['vR'][1] for s in x_all]
    y_vR3 = [s[1]['vR'][2] for s in x_all]
    y_vH1 = [s[1]['vH'][0] for s in x_all]
    y_vH2 = [s[1]['vH'][1] for s in x_all]
    ax2.plot(xs, y_vR1, label='vR1')
    ax2.plot(xs, y_vR2, label='vR2')
    ax2.plot(xs, y_vR3, label='vR3')
    ax2.plot(xs, y_vH1, label='vH1')
    ax2.plot(xs, y_vH2, label='vH2')
    plt.axvline(x=t_switch[0], color='gray', linestyle='--')
    plt.axvline(x=t_switch[1], color='gray', linestyle='dotted')
    ax2.legend(loc='lower right')

    fig3, ax3 = plt.subplots()
    y_aR1 = [s[1]['aR'][0] for s in x_all]
    y_aR2 = [s[1]['aR'][1] for s in x_all]
    y_aR3 = [s[1]['aR'][2] for s in x_all]
    y_aH1 = [s[1]['aH'][0] for s in x_all]
    y_aH2 = [s[1]['aH'][1] for s in x_all]
    ax3.plot(xs, y_aR1, label='aR1')
    ax3.plot(xs, y_aR2, label='aR2')
    ax3.plot(xs, y_aR3, label='aR3')
    ax3.plot(xs, y_aH1, label='aH1')
    ax3.plot(xs, y_aH2, label='aH2')
    plt.axvline(x=t_switch[0], color='gray', linestyle='--')
    plt.axvline(x=t_switch[1], color='gray', linestyle='dotted')
    ax3.legend(loc='upper right')

    fig4, ax4 = plt.subplots()
    y_u1 = [s[1]['aR'][0] for s in x_all]
    y_u2 = [s[1]['aR'][1] for s in x_all]
    ax4.plot(xs, y_u1, label='u1')
    ax4.plot(xs, y_u2, label='u2')
    plt.axvline(x=t_switch[0], color='gray', linestyle='--')
    plt.axvline(x=t_switch[1], color='gray', linestyle='dotted')
    ax4.legend(loc='upper right')

    # fig5, ax5 = plt.subplots()
    # y_vR1 = [s[1]['vR'][0] for s in x_all]
    # y_vH1 = [s[1]['vB'][0] for s in x_all]
    # y_dv = [r2 - r1 for r1, r2 in zip(y_vR1, y_vB1)]
    # ax5.plot(xs, y_dv, label='vR2 - vB1')
    # plt.axvline(x=t_switch, color='gray', linestyle='--')
    # plt.axhline(y=3, color='black', linestyle='--')
    # ax5.legend(loc='upper right')

    # fig4, ax4 = plt.subplots()
    # y_aH = [s[1]['aH'][0] for s in x_all]
    # y_aR = [s[1]['aR'][0] for s in x_all]
    # ax4.plot(xs, y_aH, label='aH', c='b')
    # ax4.plot(xs, y_aR, label='aR', c='r')
    # plt.axvline(x=t_switch, color='gray', linestyle='--')
    # ax4.legend(loc='upper right')

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
        # if i == 1 and t >= 2.38 and t <= 15.6:
        #     aR[i] = 2 / 1.322
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
            incentive = vel_new - fol[1] >= dvH_th - 0.1
        safety = not is_occupied(x['pH'][i], x['lH'][i] - 1)

        # dH.append(0 if fol is None 
        #           else -(incentive and safety))
        global t_switch
        if safety and t_switch[i] == -1:
            t_switch[i] = t
        dH.append(-safety)
        # dH.append(0)
        
    return (aH, dH)

def compute_u():

    lb1 = hF1.lb(x) # u1 >= lb
    ub2 = hB1.ub(x) # u2 <= ub
    lb2 = hF2.lb(x) # u2 >= lb
    ub3 = hB2.ub(x) # u3 <= ub

    A = cvxopt.matrix([[-1.0, 0.0, 0.0, 0.0], 
                       [0.0, 1.0, -1.0, 0.0], 
                       [0.0, 0.0, 0.0, 1.0]])
    b = cvxopt.matrix([-lb1, ub2, -lb2, ub3])

    print(A)
    print(b)

    u_star = [0 for _ in range(n)]
    print("BOUNDS", lb2, ub2)
    if lb2 <= ub2:
        sol = cvxopt.solvers.qp(cvxopt.matrix([[1.0, 0.0, 0.0], 
                                            [0.0, 1.0, 0.0], 
                                            [0.0, 0.0, 1.0]]), # 1.0 for min, -1.0 for max
                                cvxopt.matrix([0.0, 0.0, 0.0]), 
                                A, b)
        if sol['status'] == 'optimal':
            u_star = sol['x']
        # u_star = [-u_star[1], -u_star[0]]

    if x['lH'][0] == -1:
        u_star[0] = idm.free_road(x['vR'][0])
    if x['lH'][1] == -1:
        u_star[2] = idm.idm(x['pR'][2], x['pH'][1], x['vR'][2], x['vH'][1])
    if x['lH'][0] == -1 and x['lH'][1] == -1:
        u_star[1] = idm.idm(x['pR'][1], x['pH'][0], x['vR'][1], x['vH'][0])

    print(f"U_STAR: {u_star}")
    return (u_star, [0 for _ in range(n)])

if __name__ == '__main__':

    x['pH'].append(2*L)
    x['vH'].append(30)
    x['aH'].append(0)
    x['lH'].append(0)
    x['dH'].append(0)

    x['pH'].append(-2*L)
    x['vH'].append(30)
    x['aH'].append(0)
    x['lH'].append(0)
    x['dH'].append(0)

    x['pR'].append(2*L)
    x['vR'].append(30)
    x['aR'].append(0)
    x['lR'].append(-1)

    x['pR'].append(0)
    x['vR'].append(30)
    x['aR'].append(0)
    x['lR'].append(-1)

    x['pR'].append(-2*L)
    x['vR'].append(30)
    x['aR'].append(0)
    x['lR'].append(-1)

    hF1 = CBF_hF1_33(x, s_min, L, dt, c_min, idm)
    hB1 = CBF_hB1_33(x, s_min, L, dt, c_min, idm)
    hF2 = CBF_hF2_33(x, s_min, L, dt, c_min, idm)
    hB2 = CBF_hB2_33(x, s_min, L, dt, c_min, idm)

    u = ([0 for _ in range(n)], [0 for _ in range(n)])
    while t <= t_max:
        apply_control(compute_u())
        vis(x['pR'][1])
        t += dt

    plot()