import math
import random
from IDM import IDM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxopt
import scipy

nH = 2
nR = 1
dt = 0.1
t_max = 60
L = 5

pH = [0, 0]
vH = [0, 0]
lH = [0, 0]
pR = [0]
vR = [0]
lR = [0]

c_min = 1

s_min = 2.3
a_max = 1.45
b_max = 2.0
vR_max = 40
v_max = 30.001

j_max = 1
negj_max = 1

s0 = 1.5
v0 = 35
dvH_th = 10
T = 0.01
a = 1.8
b = 3.1
idm = IDM(s0, v0, T, a, b, L)

x_all = []
x_everything = []

alphaF = []
alphaV = []
alphaJ = []
alphanegA = []

aH_last = [0]
aR_last = [0]

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

    global x_everything

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()

    for xall in x_everything:

        xs = [s[0] for s in xall]

        y_dist = [s[1]['pR'][0] - s[1]['pH'][0] for s in xall]
        if y_dist[-1] > 20:
            continue
        ax1.plot(xs, y_dist)

        y_vH = [s[1]['vH'][0] for s in xall]
        if y_vH[-1] > 30.5:
            continue
        ax2.plot(xs, y_vH)

        y_aH = [s[1]['aH'][0] for s in xall]
        ax3.plot(xs, y_aH)

        y_u = [s[1]['aR'][0] for s in xall]
        ax4.plot(xs, y_u)

        y_jH = [(xall[i][1]['aH'][0] - xall[i - 1][1]['aH'][0]) / dt 
                for i in range(1, len(xall))]
        ax5.plot(xs[1:], y_jH)

    ax1.axhline(y=L+s_min, color='black', linestyle='--')
    ax2.axhline(y=v_max, color='black', linestyle='--')
    ax3.axhline(y=a_max, color='black', linestyle='--')
    ax3.axhline(y=-b_max, color='black', linestyle='--')
    ax5.axhline(y=j_max, color='black', linestyle='--')
    ax5.axhline(y=-negj_max, color='black', linestyle='--')

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

    x_all.append((t, {'pH': pH.copy(), 'vH': vH.copy(), 'aH': aH.copy(), 
                      'pR': pR.copy(), 'vR': vR.copy(), 'aR': aR.copy()}))

def f():
    aH = []
    dH = []
    f = get_following(0)
    aH.append(idm.free_road(vH[0]) if f == -1 
                else idm.idm(pH[0], pR[f], vH[0], vR[f]))
    aH[-1] += random.gauss(0, 0.2)
    # dH.append(0 if f == -1 
    #           else -(v0 - vR[f] >= dvH_th))
    dH.append(0)
    aH.append(random.gauss(0, 0.2))
    # aH.append(0)
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

def hA():
    return a_max - aH_last[0]
def get_alphaA():
    c = max(0, 0 / hA()) + c_min
    return [c]

def hnegA():
    return aH_last[0] + b_max
def get_alphanegA():
    c = max(0, 0 / hnegA()) + c_min
    return [c]

def compute_u():

    # Following distance lower bound
    lbF = (aR_last[0] / dt + alphaF[2] * aH_last[0] 
          + idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0]) 
          - alphaF[1] * hF_dot() - alphaF[0] * hF()) \
    / (alphaF[2] + 1 / dt - idm.df_dvR(pH[0], pR[0], vH[0], vR[0]))

    # Acceleration lower bound
    lbA = (-idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0]) 
           - alphanegA[0] * hnegA()) \
        / idm.df_dvR(pH[0], pR[0], vH[0], vR[0])

    # u df/dv + lambda + alpha h >= 0
    # u >= (-lambda - alpha h) / (df/dv)

    # Jerk lower bound
    lbJ = (-idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0]) - negj_max) \
        / idm.df_dvR(pH[0], pR[0], vH[0], vR[0])

    # u df/dv + lambda + negj_max >= 0
    # u >= (-lambda - negj_max) / (df/dv)

    lb = max(lbF, lbA, lbJ)

    # Velocity upper bound
    ubV = (alphaV[1] * hV_dot() + alphaV[0] * hV() 
          - idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0])) \
        / idm.df_dvR(pH[0], pR[0], vH[0], vR[0]) 

    # Acceleration upper bound
    ubA = (alphaA[0] * hA() 
           - idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0])) \
        / idm.df_dvR(pH[0], pR[0], vH[0], vR[0])
    
    # -u df/dv - lambda + alpha h >= 0
    # u <= (alpha h - lambda) / (df/dv)

    # Jerk upper bound
    ubJ = (j_max - idm.lamb(pH[0], pR[0], vH[0], vR[0], aH_last[0])) \
        / idm.df_dvR(pH[0], pR[0], vH[0], vR[0])

    # h = j_max - u df/dv - lambda >= 0
    # u <= (j_max - lambda) / (df/dv)

    ub = min(ubV, ubA, ubJ)

    u_star = 0
    if lb <= ub:
        sol = cvxopt.solvers.qp(cvxopt.matrix([[1.0]]), # 1.0 for min, -1.0 for max
                                cvxopt.matrix([0.0]), 
                                cvxopt.matrix([[-1.0, 1.0]]), 
                                cvxopt.matrix([-lb, ub]))
        u_star = sol['x'][0]
        # print(sol)
    # else:
        # print("INFEASIBLE")

    # J(u) = ||u||^2
    # u_star = max(0, lb)
    # u_star = min(0, ub)

    # print(f"U_STAR: {u_star}")
    return ([u_star for _ in range(nR)], 
            [0 for _ in range(nR)])

def compute_u_idm():
    # return ([idm.idm(pR[0], pH[1], vR[0], vH[1]) + random.gauss(0, 0.2)], [0])
    return ([idm.idm(pR[0], pH[1], vR[0], vH[1])], [0])

if __name__ == '__main__':

    avg_jerk_opt = []
    avg_jerk_non = []

    for iter in range(100):

        print("ITER: ", iter)

        s0 += random.gauss(0, 0.2)
        v0 += random.gauss(0, 2)
        a += random.gauss(0, 0.1)
        b += random.gauss(0, 0.1)

        # pH[0] = -2.001*L
        # vH[0] = 30
        # lH[0] = 0

        # pR[0] = 0
        # vR[0] = 30
        # lR[0] = 0

        # pH[1] = 2*L
        # vH[1] = 30
        # lH[1] = 0

        pH[0] = random.uniform(-3*L, -1*L)
        vH[0] = random.uniform(25, 35)
        lH[0] = 0

        pR[0] = 0
        vR[0] = random.uniform(25, 35)
        lR[0] = 0

        pH[1] = random.uniform(1*L, 3*L)
        vH[1] = random.uniform(25, 35)
        lH[1] = 0

        alphaF = get_alphaF()
        alphaV = get_alphaV()
        alphaA = get_alphaA()
        alphanegA = get_alphanegA()

        t = 0
        u = ([0 for _ in range(nR)], [0 for _ in range(nR)])
        x_all = []
        while t <= t_max:
            apply_control(compute_u_idm())
            # vis(min(pH[0] + s_min, pR[0]))
            t += dt
        
        y_jH = [(x_all[i][1]['aH'][0] - x_all[i - 1][1]['aH'][0]) / dt 
            for i in range(1, len(x_all))]
        # print(x_all)
        avg_jerk_non.append(sum([abs(j) for j in y_jH]) / (t_max / dt - 1))

        
        pH[0] = random.uniform(-3*L, -1*L)
        vH[0] = random.uniform(25, 35)
        lH[0] = 0

        pR[0] = 0
        vR[0] = random.uniform(25, 35)
        lR[0] = 0

        pH[1] = random.uniform(1*L, 3*L)
        vH[1] = random.uniform(25, 35)
        lH[1] = 0

        alphaF = get_alphaF()
        alphaV = get_alphaV()
        alphaA = get_alphaA()
        alphanegA = get_alphanegA()

        t = 0
        u = ([0 for _ in range(nR)], [0 for _ in range(nR)])
        x_all = []
        while t <= t_max:
            apply_control(compute_u())
            # vis(min(pH[0] + s_min, pR[0]))
            t += dt

        if not safe():
            iter -= 1
            avg_jerk_non.pop()
            continue

        x_everything.append(x_all)

        y_jH = [(x_all[i][1]['aH'][0] - x_all[i - 1][1]['aH'][0]) / dt 
            for i in range(1, len(x_all))]
        avg_jerk_opt.append(sum([abs(j) for j in y_jH]) / (t_max / dt - 1))

    print(avg_jerk_non)
    print(avg_jerk_opt)

    print("AVG OF NON: ", sum(avg_jerk_non) / len(avg_jerk_non))
    print("AVG OF OPT: ", sum(avg_jerk_opt) / len(avg_jerk_opt))

    ttest1 = scipy.stats.ttest_rel(avg_jerk_non, avg_jerk_opt)
    print(ttest1)

    plot()
