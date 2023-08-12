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

pH = []
vH = []
lH = []
pR = []
vR = []
lR = []

aH_last = 0
aR_last = 0

s_min = 1.5*L
a_max = 2.0
b_max = 4.0
vR_max = 35
p_min = 0.1

s0 = 0.3*L
v0 = 30 # 30
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
        
    if lH[0] != -1:
        ax.add_patch(Rectangle(
                (-1*L - L/4, pH[0] - dist_min / 2), 
                L/2, dist_min,
                facecolor=(1,1,0,0.5)))

    plt.pause(0.001)

def safe(): 
    return pR[0] - pH[0] - L >= s_min or lH[0] != lR[0]

def get_following(i):
    f = -1
    for j in range(nH):
        if i == j:
            continue
        if lH[i] != lH[j] or pH[i] > pH[j]:
            continue
        if f == -1 or pH[j] < pH[f]:
            f = j
    if f != -1:
        return pH[f], vH[f]
    for j in range(nR):
        if lH[i] != lR[j] or pH[i] > pR[j]:
            continue
        if f == -1 or pR[j] < pR[f]:
            f = j
    if f != -1:
        return pR[f], vR[f]
    return None

def get_new_following(i, l):
    f = -1
    for j in range(nH):
        if lH[j] != l or pH[i] > pH[j]:
            continue
        if f == -1 or pH[j] < pH[f]:
            f = j
    if f != -1:
        return pH[f], vH[f]
    for j in range(nR):
        if lR[j] != l or pH[i] > pR[j]:
            continue
        if f == -1 or pR[j] < pR[f]:
            f = j
    if f != -1:
        return pR[f], vR[f]
    return None

def apply_control(u):

    aH, dH = f()
    aR, dR = u

    for i in range(nH):
        vH[i] += min(a_max, max(-b_max, aH[i])) * dt
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
    aH_last = aH[0]
    aR_last = aR[0]

def is_occupied(x, l):
    for i in range(nH):
        if lH[i] == l and abs(x - pH[i]) < dist_min:
            return True
    for i in range(nR):
        if lR[i] == l and abs(x - pR[i]) < dist_min:
            return True
    return False

def f():

    aH = []
    dH = []

    for i in range(nH):
        if const_vel[i]:
            fol = get_following(i)
            if fol == None:
                aH.append(0)
            else:
                aH.append(idm.idm(pH[i], fol[0], vH[i], fol[1]))
            dH.append(0)
            continue
        
        f = get_following(i)
        aH.append(idm.free_road(vH[i]) if f is None 
                  else idm.idm(pH[i], f[0], vH[i], f[1]))
        
        incentive = False
        if f is not None:
            f_new = get_new_following(i, -1)
            vel_new = v0 if f_new is None else f_new[1]
            incentive = vel_new - f[1] >= dvH_th

        safety = not is_occupied(pH[i], lH[i] - 1)
        dH.append(0 if f is None 
                  else -(incentive and safety))
        
    return (aH, dH)

def hS_b():
    return pH[1] - pH[0] - dist_min
def hS_b_dot():
    return vH[1] - vH[0]
def hS_b_ddot():
    return -aH_last
def hS_b_dddot():
    return -idm.df_dvR(pH[0], pR[0], vH[0], vR[0], 
                        aH_last, aR_last) * aR_last
def alpha_beta_gamma_b():
    p1 = max(0, -hS_b_dot() / hS_b()) + p_min
    p2 = max(0, -(hS_b_ddot() + p1 * hS_b_dot()) 
                    / (hS_b_dot() + p1 * hS_b())) + p_min
    p3 = max(0, -(hS_b_dddot() + p2 * hS_b_ddot() + p1 * hS_b_dot()) 
                    / (hS_b_ddot() + p2 * hS_b_dot() + p1 * hS_b())) + p_min
    return (p1 + p2 + p3, p1*p2 + p1*p3 + p2*p3, p1 * p2 * p3)

def hS_f():
    return pH[0] - pH[1] - dist_min
def hS_f_dot():
    return vH[0] - vH[1]
def hS_f_ddot():
    return aH_last
def hS_f_dddot():
    return idm.df_dvR(pH[0], pR[0], vH[0], vR[0], 
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

    solutions = []
    u_star_b, u_star_f = np.inf, np.inf

    df_dvR = idm.df_dvR(pH[0], pR[0], vH[0], vR[0], aH_last, aR_last)
    control_lb = idm.control_lower_bound(b_max, b_max, 0.1, 
                                         pH[0], pR[0], vH[0], vR[0], dt)

    phiS_b = 1 / df_dvR * (alpha_b * hS_b_ddot() + 
                             beta_b * hS_b_dot() + gamma_b * hS_b()) # u <= phiS_b
    print(f"+++++++++++++++ {hS_b_ddot()} {hS_b_dot()} {hS_b()}")
    phiI_b = p4_b * hI_b() # u <= phiI_b
    lb_b = max(-b_max, control_lb)
    ub_b = min(a_max, min(phiS_b, phiI_b))
    print(f"@@@@@@@@@@@ {phiS_b} {df_dvR} {phiI_b} {lb_b} {ub_b}")

    if lb_b <= ub_b:
        sol_b = cvxopt.solvers.qp(cvxopt.matrix([[-1.0]]), # 1.0 for min, -1.0 for max
                                cvxopt.matrix([0.0]), 
                                cvxopt.matrix([[-1.0, 1.0]]), 
                                cvxopt.matrix([-lb_b, ub_b]))
        print(f"phiS_b: {phiS_b} phiI_b: {phiI_b}")
        print(sol_b)
        u_star_b = sol_b['x'][0]
        solutions.append(u_star_b)
        print(f'~~~~~~~u_star_b = {u_star_b}~~~~~~~')

    # phiS_f = -1 / df_dvR * (alpha_f * hS_f_ddot() + 
    #                           beta_f * hS_f_dot() + gamma_f * hS_f()) # u >= phiS_f
    # print(f"############ {phiS_f} {df_dvR} {hS_f_ddot()} {hS_f_dot()} {hS_f()}")
    # print(alpha_f, beta_f, gamma_f)
    # phiI_f = p4_f * hI_f() # u <= phiI_f
    # lb_f = max(-b_max, max(control_lb, phiS_f))
    # ub_f = min(a_max, phiI_f)

    # if lb_f <= ub_f:
    #     sol_f = cvxopt.solvers.qp(cvxopt.matrix([[1.0]]), # 1.0 for min, -1.0 for max
    #                             cvxopt.matrix([0.0]), 
    #                             cvxopt.matrix([[-1.0, 1.0]]), 
    #                             cvxopt.matrix([-lb_f, ub_f]))
    #     print(f"phiS_f: {phiS_f} phiI_f: {phiI_f}")
    #     print(sol_f)
    #     u_star_f = sol_f['x'][0]
    #     solutions.append(u_star_f)
    #     print(f'~~~~~~~u_star_f = {u_star_f}~~~~~~~')

    u_star = np.inf
    for sol in solutions:
        if sol**2 < u_star**2:
            u_star = sol

    if u_star == u_star_b:
        print(f"U_STAR = U_STAR_B = {u_star}")
    else:
        print(f"U_STAR = U_STAR_F = {u_star}")

    return ([u_star for _ in range(nR)], 
            [0 for _ in range(nR)])

if __name__ == '__main__':

    pH.append(-2*L)
    vH.append(26)
    lH.append(0)

    pR.append(0)
    vR.append(26)
    lR.append(0)

    pH.append(-4*L)
    vH.append(25)
    lH.append(-1)

    u_init = 0.1
    apply_control(([u_init for _ in range(nR)], [0 for _ in range(nR)]))

    alpha_b, beta_b, gamma_b = alpha_beta_gamma_b()
    alpha_f, beta_f, gamma_f = alpha_beta_gamma_b()
    p4_b = get_p4_b()
    p4_f = get_p4_f()
    
    was_safe = True
    u = ([0 for _ in range(nR)], [0 for _ in range(nR)])
    while t <= t_max:
        # if safe() and not was_safe:
        #     u = ([2 if vR[0] < 20 else 0], [0])
        # if was_safe and not safe():
        #     u = compute_u()
        # was_safe = safe()
        # apply_control(u)
        apply_control(compute_u())
        plot(min(pH[0] + s_min, pR[0]))
        t += dt
