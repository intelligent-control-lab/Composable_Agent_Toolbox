import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

nH = 1
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

s_min = 1.4*L
a_max = 1.5
vR_max = 35

s0 = 0.3*L
v0 = 35
dvH_th = 10
T = 0.01 # 1.3
a = 1.8
b = 3.1

alpha = 0
beta = 0
p3 = 0

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
    ax.add_patch(Rectangle(
            (lR[i]*L - L/4, xR[i] - L/2 - s_min), 
            L/2, s_min,
            facecolor=(1,1,0,0.5)))

    plt.pause(0.001)

def safe(): 
    return xR[0] - xH[0] - L >= s_min or lH[0] != lR[0]

def get_following(i):
    f = -1
    for j in range(nR):
        if lH[i] != lR[j] or xH[i] > xR[j]:
            continue
        if f == -1 or xR[j] < xR[f]:
            f = j
    return f

def apply_control(u):

    aH, dH = f_theta()
    aR, dR = u

    for i in range(nH):
        vH[i] += aH[i] * dt
        xH[i] += vH[i] * dt
        lH[i] = max(-1, min(1, 
            lH[i] + round(dH[i])))
        
    for i in range(nR):
        aR[i] = min(a_max, aR[i])
        vR[i] += aR[i] * dt
        vR[i] = min(vR_max, vR[i])
        xR[i] += vR[i] * dt
        lR[i] = max(-1, min(1, 
            lR[i] + round(dR[i])))

def f_theta():
    aH = []
    dH = []
    for i in range(nH):
        f = get_following(i)
        acc = a * (1 - (vH[i] / v0)**4)
        if f != -1:
            s_star = s0 + vH[i] * T + (vH[i] * (vH[i] - vR[f])) / (2 * math.sqrt(a * b))
            acc += -a * (s_star / (xR[f] - xH[i] - L))**2
        aH.append(acc)
        dH.append(0 if f == -1 else v0 - vR[f] >= dvH_th)
    return (aH, dH)

def hF():
    return xR[0] - xH[0] - L - s_min
def hF_dot():
    return vR[0] - vH[0]
def hF_ddot():
    return compute_u()[0][0] - f_theta()[0][0]
def get_alphabeta():
    p1 = max(0, -hF_dot() / hF()) + 0.1
    p2 = max(0, -(hF_ddot() + p1 * hF_dot()) 
                    / (hF_dot() + p1 * hF())) + 0.1
    return (p1 + p2, p1 * p2)

def hL():
    return v0 - vR[0] - dvH_th
def hL_dot():
    return vR[0] - vH[0]
def get_p3():
    return max(0, -hL_dot() / hL()) + 0.1

def compute_u():
    phiF = f_theta()[0][0] - alpha * hF_dot() - beta * hF()
    if v0 > vR_max:
        phiF = np.inf
    uF = max(0, phiF)
    uL = min(0, p3 * hL())
    u_star = uF if uF**2 < uL**2 else uL
    return ([u_star for _ in range(nR)], 
            [0 for _ in range(nR)])

if __name__ == '__main__':

    for i in range(nH):
        xH.append(-2*L)
        vH.append(30)
        lH.append(0)
    for i in range(nR):
        xR.append(0)
        vR.append(26)
        lR.append(0)

    alpha, beta = get_alphabeta()
    p3 = get_p3()

    was_safe = True
    u = ([0], [0])
    while t <= t_max:
        if was_safe and not safe():
            u = compute_u()
        if safe() and not was_safe:
            u = ([0], [0])
        was_safe = safe()
        apply_control(u)
        plot(min(xH[0] + s_min, xR[0]))
        t += dt