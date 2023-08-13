import math
import random
from IDM import IDM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxopt
import scipy

nH = 1
nR = 1
dt = 0.1
t_max = 60
L = 5

pH = []
vH = []
lH = []
pR = []
vR = []
lR = []

c_min = 1

s_min = 10.0001 - L
a_max = 2.0
b_max = 4.0
vR_max = 40
v_max = 25

s0 = 0.3*L
v0 = 36
dvH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1
idm = IDM(s0, v0, T, a, b, L)

alphaF = []
alphaV = []

aH_last = [0]
aR_last = [0]

state_all = []

t = 0
safe = False

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
    for i in range(nH):
        ax.text(lH[i]*L, pH[i], str(vH[i]))
    for i in range(nR):
        ax.text(lR[i]*L, pR[i], str(vR[i]))

    for i in range(nH):
        ax.add_patch(Rectangle(
            (lH[i]*L - L/4, pH[i] - L/2), 
            L/2, L, 
            facecolor='blue'))
    for i in range(nR):
        ax.add_patch(Rectangle(
            (lR[i]*L - L/4, pR[i] - L/2), 
            L/2, L, 
            facecolor='red'))
    ax.add_patch(Rectangle(
            (lR[i]*L - L/4, pR[i] - L/2 - s_min), 
            L/2, s_min,
            facecolor=(1,1,0,0.5)))

    plt.pause(0.001)

def plot():

    global state_all
    state_all = state_all[10:]

    x = [s[0] for s in state_all]

    fig0, ax0 = plt.subplots()
    y_pH = [s[1][0] for s in state_all]
    y_pR = [s[2][0] for s in state_all]
    y = [r - h for h, r in zip(y_pH, y_pR)]
    ax0.plot(x, y, label='dist')
    # plt.axhline(y=s_min+L, color='black', linestyle='--')
    # ax0.plot(x, y_pH, label='pH', c='b')
    # ax0.plot(x, y_pR, label='pR', c='r')
    ax0.legend(loc='upper right')

    fig1, ax1 = plt.subplots()
    y = [s[3][0] for s in state_all]
    ax1.plot(x, y, label='vH')
    plt.axhline(y=v_max, color='black', linestyle='--')
    ax1.legend(loc='upper right')

    fig2, ax2 = plt.subplots()
    y_vH = [s[3][0] for s in state_all]
    y_vR = [s[4][0] for s in state_all]
    ax2.plot(x, y_vH, label='vH', c='b')
    ax2.plot(x, y_vR, label='vR', c='r')
    ax2.legend(loc='upper right')

    fig3, ax3 = plt.subplots()
    y_u = [s[6][0] for s in state_all]
    ax3.plot(x, y_u, label='u')
    ax3.legend(loc='lower right')

    fig4, ax4 = plt.subplots()
    y_aH = [s[5][0] for s in state_all]
    y_aR = [s[6][0] for s in state_all]
    ax4.plot(x, y_aH, label='aH', c='b')
    ax4.plot(x, y_aR, label='aR', c='r')
    ax4.legend(loc='lower right')

    plt.show()

def safe(): 
    return pR[0] - pH[0] - L >= s_min or lH[0] != lR[0]

def get_following(i):
    f = -1
    for j in range(nR):
        if lH[i] != lR[j] or pH[i] > pR[j]:
            continue
        if f == -1 or pR[j] < pR[f]:
            f = j
    return f

def apply_control(u):

    aH, dH = f()
    aR, dR = u

    for i in range(nH):
        aH[i] = min(a_max, max(-b_max, aH[i]))
        vH[i] += aH[i] * dt
        pH[i] += vH[i] * dt
        lH[i] = max(-1, min(1, 
            lH[i] + round(dH[i])))
        
    for i in range(nR):
        aR[i] = min(a_max, max(-b_max, aR[i]))
        vR[i] += aR[i] * dt
        vR[i] = min(vR_max, vR[i])
        pR[i] += vR[i] * dt
        lR[i] = max(-1, min(1, 
            lR[i] + round(dR[i])))
        
    global aH_last, aR_last
    aH_last = aH
    aR_last = aR
    state_all.append([t, pH.copy(), pR.copy(), vH.copy(), vR.copy(), aH.copy(), aR.copy()])

def f():
    aH = []
    dH = []
    for i in range(nH):
        f = get_following(i)
        aH.append(idm.free_road(vH[i]) if f == -1 
                  else idm.idm(pH[i], pR[f], vH[i], vR[f]))
        # dH.append(0 if f == -1 
        #           else -(v0 - vR[f] >= dvH_th))
        dH.append(0)
    return (aH, dH)

def hF():
    return pR[0] - pH[0] - L - s_min
def hF_dot():
    return vR[0] - vH[0]
def hF_ddot():
    return aR_last[0] - aH_last[0]
def get_alphaF():
    c1 = max(0, -hF_dot() / hF()) + c_min
    c2 = max(0, -(hF_ddot() + c1 * hF_dot()) 
                    / (hF_dot() + c1 * hF())) + c_min
    c3 = max(0, -(0 + c2 * hF_ddot() + c1 * hF_dot()) 
             / (hF_ddot() + c2 * hF_dot() + c1 * hF())) + c_min
    return [c1 + c2 + c3, 
            c1*c2 + c1*c3 + c2*c3,
            c1 * c2 * c3]

def hV():
    return v_max - vH[0]
def hV_dot():
    return -aH_last[0]
def get_alphaV():
    c1 = max(0, -hV_dot() / hV()) + c_min
    c2 = max(0, -(0 + c1 * hV_dot()) 
             / (hV_dot() + c1 * hV())) + c_min
    return [c1 + c2, c1 * c2]

# def hL():
#     return v0 - vR[0] - dvH_th
# def hL_dot():
#     return 0
# def get_p3():
#     return max(0, -hL_dot() / hL()) + p_min

def compute_u():

    # Following distance lower bound
    lb = (aR_last[0] / dt + alphaF[2] * f()[0][0] 
          + idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0]) 
          - alphaF[1] * hF_dot() - alphaF[0] * hF()) \
    / (alphaF[2] + 1 / dt - idm.df_dvR(pH[0], pR[0], vH[0], vR[0])) # u >= lb

    # Velocity upper bound
    ub = (alphaV[1] * hV_dot() + alphaV[0] * hV() 
          - idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0])) \
    / idm.df_dvR(pH[0], pR[0], vH[0], vR[0]) # u <= ub

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
    u_star = min(0, ub)

    print(f"U_STAR: {u_star}")
    return ([u_star for _ in range(nR)], 
            [0 for _ in range(nR)])

if __name__ == '__main__':

    pH.append(-2*L)
    vH.append(30)
    lH.append(0)

    pR.append(0)
    vR.append(26)
    lR.append(0)

    alphaF = get_alphaF()
    alphaV = get_alphaV()

    u = ([0 for _ in range(nR)], [0 for _ in range(nR)])
    while t <= t_max:
        apply_control(compute_u())
        vis(min(pH[0] + s_min, pR[0]))
        t += dt

    plot()
