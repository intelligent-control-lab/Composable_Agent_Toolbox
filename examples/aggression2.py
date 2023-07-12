import math
import random
from IDM import IDM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cvxopt

nH = 2
nR = 1

dt = 0.1
t_max = 100000
L = 5

xH = []
vH = []
lH = []
xR = []
vR = []
lR = []

aH_last = 0
aR_last = 0

s_min = 1.4*L
a_max = 2.0
b_max = 4.0
vR_max = 35
p_min = 0.1

s0 = 0.3*L
v0 = 35
dvH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1
idm = IDM(s0, v0, T, a, b, L)

dist_min = 1.5 * L

const_vel = [False, True]

alpha_b = 0
beta_b = 0
gamma_b = 0
p4_b = 0

alpha_f = 0
beta_f = 0
gamma_f = 0
p4_f = 0

t = 0
safe = False

fig, ax = plt.subplots()

def plot(pov):
    
    ax.cla()
    ax.axis([-4*L, 4*L, pov - 4*L, pov + 4*L])
    ax.axvline(x=-L/2, color="black", linestyle="--")
    ax.axvline(x=L/2, color="black", linestyle="--")
    ax.axvline(x=-3*L/2, color="black", linestyle="-")
    ax.axvline(x=3*L/2, color="black", linestyle="-")

    ax.text(-4*L, pov - 4*L, "Safe: " + str(safe()))
    ax.text(-4*L, pov + 4*L, "t = " + str(t))
    for i in range(nH):
        ax.text(lH[i]*L, xH[i], str(vH[i]))
    for i in range(nR):
        ax.text(lR[i]*L, xR[i], str(vR[i]))

    for i in range(nH):
        ax.add_patch(Rectangle(
            (lH[i]*L - L/4, xH[i] - L/2), 
            L/2, L, 
            facecolor='blue'))
    for i in range(nR):
        ax.add_patch(Rectangle(
            (lR[i]*L - L/4, xR[i] - L/2), 
            L/2, L, 
            facecolor='red'))
        
    if lH[0] != -1:
        ax.add_patch(Rectangle(
                (-1*L - L/4, xH[0] - dist_min / 2), 
                L/2, dist_min,
                facecolor=(1,1,0,0.5)))

    plt.pause(0.001)

def safe(): 
    return xR[0] - xH[0] - L >= s_min or lH[0] != lR[0]

def get_following(i):
    f = -1
    for j in range(nH):
        if i == j:
            continue
        if lH[i] != lH[j] or xH[i] > xH[j]:
            continue
        if f == -1 or xH[j] < xH[f]:
            f = j
    if f != -1:
        return xH[f], vH[f]
    for j in range(nR):
        if lH[i] != lR[j] or xH[i] > xR[j]:
            continue
        if f == -1 or xR[j] < xR[f]:
            f = j
    if f != -1:
        return xR[f], vR[f]
    return None

def get_new_following(i, l):
    f = -1
    for j in range(nH):
        if lH[j] != l or xH[i] > xH[j]:
            continue
        if f == -1 or xH[j] < xH[f]:
            f = j
    if f != -1:
        return xH[f], vH[f]
    for j in range(nR):
        if lR[j] != l or xH[i] > xR[j]:
            continue
        if f == -1 or xR[j] < xR[f]:
            f = j
    if f != -1:
        return xR[f], vR[f]
    return None

def apply_control(u):

    aH, dH = f_theta()
    aR, dR = u

    for i in range(nH):
        vH[i] += min(a_max, max(-b_max, aH[i])) * dt
        xH[i] += vH[i] * dt
        lH[i] = max(-1, min(1, 
            lH[i] + round(dH[i])))
        
    for i in range(nR):
        aR[i] = min(a_max, max(-b_max, aR[i]))
        vR[i] += aR[i] * dt
        vR[i] = min(vR_max, vR[i])
        xR[i] += vR[i] * dt
        lR[i] = max(-1, min(1, 
            lR[i] + round(dR[i])))
        
    aH_last = aH[0]
    aR_last = aR[0]

def is_occupied(x, l):
    for i in range(nH):
        if lH[i] == l and abs(x - xH[i]) < dist_min:
            return True
    for i in range(nR):
        if lR[i] == l and abs(x - xR[i]) < dist_min:
            return True
    return False

def f_theta():

    aH = []
    dH = []

    for i in range(nH):
        if const_vel[i]:
            aH.append(0)
            dH.append(0)
            continue
        
        f = get_following(i)
        aH.append(idm.free_road(vH[i]) if f is None 
                  else idm.idm(xH[i], f[0], vH[i], f[1]))
        
        incentive = False
        if f is not None:
            f_new = get_new_following(i, -1)
            vel_new = v0 if f_new is None else f_new[1]
            incentive = vel_new - f[1] >= dvH_th

        safety = not is_occupied(xH[i], lH[i] - 1)
        dH.append(0 if f is None 
                  else -(incentive and safety))
        
    return (aH, dH)

def hS_b():
    return xH[1] - xH[0] - dist_min
def hS_b_dot():
    return vH[1] - vH[0]
def hS_b_ddot():
    return -aH_last
def hS_b_dddot():
    return -idm.dIdm_dvR(xH[0], xR[0], vH[0], vR[0], 
                        aH_last, aR_last) * aR_last
def alpha_beta_gamma_b():
    p1 = max(0, -hS_b_dot() / hS_b()) + p_min
    p2 = max(0, -(hS_b_ddot() + p1 * hS_b_dot()) 
                    / (hS_b_dot() + p1 * hS_b())) + p_min
    p3 = max(0, -(hS_b_dddot() + p2 * hS_b_ddot() + p1 * hS_b_dot()) 
                    / (hS_b_ddot() + p2 * hS_b_dot() + p1 * hS_b())) + p_min
    return (p1 + p2 + p3, p1*p2 + p1*p3 + p2*p3, p1 * p2 * p3)

def hS_f():
    return xH[0] - xH[1] - dist_min
def hS_f_dot():
    return vH[0] - vH[1]
def hS_f_ddot():
    return aH_last
def hS_f_dddot():
    return idm.dIdm_dvR(xH[0], xR[0], vH[0], vR[0], 
                        aH_last, aR_last) * aR_last
def alpha_beta_gamma_f():
    p1 = max(0, -hS_f_dot() / hS_f()) + p_min
    p2 = max(0, -(hS_f_ddot() + p1 * hS_f_dot()) 
                    / (hS_f_dot() + p1 * hS_f())) + p_min
    p3 = max(0, -(hS_f_dddot() + p2 * hS_f_ddot() + p1 * hS_f_dot()) 
                    / (hS_f_ddot() + p2 * hS_f_dot() + p1 * hS_f())) + p_min
    return (p1 + p2 + p3, p1*p2 + p1*p3 + p2*p3, p1 * p2 * p3)

def hI_b():
    return vH[1] - vR[0] - dvH_th
def hI_b_dot():
    return -aR_last
def get_p4_b():
    return max(0, -hI_b_dot() / hI_b()) + p_min

def hI_f():
    return v0 - vR[0] - dvH_th
def hI_f_dot():
    return -aR_last
def get_p4_f():
    return max(0, -hI_f_dot() / hI_f()) + p_min

def compute_u():

    phiS_b = idm.dIdm_dvR(xH[0], xR[0], vH[0], vR[0], aH_last, aR_last) \
        * (alpha_b * hS_b_ddot() + beta_b * hS_b_dot() + gamma_b * hS_b()) # u <= phiS_b
    phiI_b = p4_b * hI_b() # u <= phiI_b
    sol_b = cvxopt.solvers.qp(cvxopt.matrix([[1.0]]), 
                            cvxopt.matrix([0.0]), 
                            cvxopt.matrix([[1.0, 1.0]]), 
                            cvxopt.matrix([phiS_b, phiI_b]))
    print(sol_b)
    u_star_b = sol_b['x'][0]
    print(f'~~~~~~~u_star_b = {u_star_b}~~~~~~~')

    phiS_f = idm.dIdm_dvR(xH[0], xR[0], vH[0], vR[0], aH_last, aR_last) \
        * (alpha_f * hS_f_ddot() + beta_f * hS_f_dot() + gamma_f * hS_f()) # u >= phiS_f
    phiI_f = p4_f * hI_f() # u <= phiI_f
    print(phiS_f, phiI_f)
    u_star_f = np.inf
    if phiS_f <= phiI_f:
        sol_f = cvxopt.solvers.qp(cvxopt.matrix([[1.0]]), 
                                cvxopt.matrix([0.0]), 
                                cvxopt.matrix([[-1.0, 1.0]]), 
                                cvxopt.matrix([-phiS_f, phiI_f]))
        print(sol_f)
        u_star_f = sol_f['x'][0]
        print(f'~~~~~~~u_star_f = {u_star_f}~~~~~~~')

    u_star = u_star_b if u_star_b**2 < u_star_f**2 else u_star_f
    return ([u_star for _ in range(nR)], 
            [0 for _ in range(nR)])

if __name__ == '__main__':

    xH.append(-2*L)
    vH.append(30)
    lH.append(0)

    xR.append(0)
    vR.append(26)
    lR.append(0)

    xH.append(-3*L)
    vH.append(24)
    lH.append(-1)

    alpha_b, beta_b, gamma_b = alpha_beta_gamma_b()
    alpha_f, beta_f, gamma_f = alpha_beta_gamma_b()
    p4_b = get_p4_b()
    p4_f = get_p4_f()
    
    was_safe = True
    u = ([0 for _ in range(nR)], [0 for _ in range(nR)])
    while t <= t_max:
        if safe() and not was_safe:
            u = ([2 if vR[0] < 20 else 0], [0])
        if was_safe and not safe():
            u = compute_u()
        was_safe = safe()
        apply_control(u)
        plot(min(xH[0] + s_min, xR[0]))
        t += dt
