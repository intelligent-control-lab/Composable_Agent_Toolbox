import math
import random
import cvxopt
from IDM import IDM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

nH = 1
nR = 1
dt = 0.1
t_max = 100000
L = 5

pH = []
vH = []
lH = []
pR = []
vR = []
lR = []

aH_last = 0
aH_last2 = 0
aR_last = 0

s_min = 0.5*L
v_max = 200
a_max = 1
j_max = 100

vR_max = 35
aR_max = 1.5
p_min = 0.1

s0 = 0.3*L
v0 = 35
dvH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1
idm = IDM(s0, v0, T, a, b, L)

alpha_F = 0
beta_F = 0
alpha_V = 0 
beta_V = 0
p_A = 0

t = 0

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots(2, 2)

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

    ax2[0, 0].set_title("Following Distance")
    ax2[0, 1].set_title("Velocity")
    ax2[1, 0].set_title("Acceleration")
    ax2[1, 1].set_title("Jerk")

    ax2[0, 0].scatter(t, pR[0] - pH[0], color='b')
    ax2[0, 1].scatter(t, vH[0], color='b')
    ax2[1, 0].scatter(t, aH_last, color='b')
    global aH_last2
    ax2[1, 1].scatter(t, (aH_last - aH_last2) / dt, color='b')
    aH_last2 = aH_last

    plt.pause(0.001)

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
        vH[i] += aH[i] * dt
        pH[i] += vH[i] * dt
        lH[i] = max(-1, min(1, 
            lH[i] + round(dH[i])))
        
    for i in range(nR):
        aR[i] = min(aR_max, aR[i])
        vR[i] += aR[i] * dt
        vR[i] = min(vR_max, vR[i])
        pR[i] += vR[i] * dt
        lR[i] = max(-1, min(1, 
            lR[i] + round(dR[i])))
        
    global aH_last, aR_last
    aH_last = aH[0]
    aR_last = aR[0]

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
    return compute_u()[0][0] - f()[0][0]
def get_alphabetaF():
    p1 = max(0, -hF_dot() / hF()) + p_min
    p2 = max(0, -(hF_ddot() + p1 * hF_dot()) 
                    / (hF_dot() + p1 * hF())) + p_min
    return (p1 + p2, p1 * p2)

def hV():
    return v_max - vH[0]
def hV_dot():
    return -aH_last
def hV_ddot():
    return -aR_last * idm.df_dvR(pH[0], pR[0], vH[0], vR[0], 
                        aH_last, aR_last)
def get_alphabetaV():
    p1 = max(0, -hV_dot() / hV()) + p_min
    p2 = max(0, -(hV_ddot() + p1 * hV_dot()) 
                    / (hV_dot() + p1 * hV())) + p_min
    return (p1 + p2, p1 * p2)

def hA():
    return a_max - aH_last
def hA_dot():
    return -aR_last * idm.df_dvR(pH[0], pR[0], vH[0], vR[0], 
                        aH_last, aR_last)
def get_pA():
    return max(0, -hA_dot() / hA()) + p_min

def hJ():
    return j_max - aR_last * idm.df_dvR(pH[0], pR[0], vH[0], vR[0], 
                        aH_last, aR_last)

def compute_u():

    lb_F = aH_last - alpha_F * hF_dot() - beta_F * hF()
    dvR_df = 1 / idm.df_dvR(pH[0], pR[0], vH[0], vR[0], 
                            aH_last, aR_last)
    ub_V = (alpha_V * hV_dot() + beta_V * hV()) * dvR_df
    ub_A = p_A * hA() * dvR_df
    ub_J = j_max * dvR_df

    lb = lb_F
    # lb = -b if safe() else lb_F
    ub = min(ub_V, ub_A, ub_J)

    print(f"LB: {lb}\tUB: {ub}")

    u_star = 0
    if lb <= ub:
        sol = cvxopt.solvers.qp(cvxopt.matrix([[1.0]]), # 1.0 for min, -1.0 for max
                                cvxopt.matrix([0.0]), 
                                cvxopt.matrix([[-1.0, 1.0]]), 
                                cvxopt.matrix([-lb, ub]))
        print(sol)
        u_star = sol['x'][0]
    else:
        print("INFEASIBLE!")

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

    u_init = 0.1
    apply_control(([u_init for _ in range(nR)], [0 for _ in range(nR)]))

    alpha_F, beta_F = get_alphabetaF()
    alpha_V, beta_V = get_alphabetaV()
    p_A = get_pA()

    u = ([0 for _ in range(nR)], [0 for _ in range(nR)])
    while t <= t_max:
        apply_control(compute_u())
        vis(min(pH[0] + s_min, pR[0]))
        plot()
        t += dt
