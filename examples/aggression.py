import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

nH = 1
nR = 1
dt = 0.01
T = 10000
L = 1

xH = []
vH = []
lH = []
xR = []
vR = []
lR = []

s_min = L
vH_max = [30]
dvH_th = [10]

alpha = 0
beta = 0
p3 = 0

fig, ax = plt.subplots()

def plot(fref):
    
    ax.cla()
    ax.axis([-4*L, 4*L, fref - 4*L, fref + 4*L])
    ax.axvline(x=-L/2, color="black", linestyle="--")
    ax.axvline(x=L/2, color="black", linestyle="--")
    ax.axvline(x=-3*L/2, color="black", linestyle="-")
    ax.axvline(x=3*L/2, color="black", linestyle="-")

    for i in range(nH):
        ax.add_patch(Rectangle(
            (lH[i] - L/4, xH[i] - L/2), 
            L/2, L, 
            facecolor='blue'))
    for i in range(nR):
        ax.add_patch(Rectangle(
            (lR[i] - L/4, xR[i] - L/2), 
            L/2, L, 
            facecolor='red'))

    plt.pause(0.001)

def apply_control(u):

    aH, dH = f_theta()
    aR, dR = u

    f_theta_last = [(aH, dH)]
    u_last = [(aR, dR)]

    for i in range(nH):
        vH[i] += aH[i] * dt
        xH[i] += vH[i] * dt
        lH[i] = max(-1, min(1, 
            lH[i] + round(dH[i])))
        
    for i in range(nR):
        vR[i] += aR[i] * dt
        xR[i] += vR[i] * dt
        lR[i] = max(-1, min(1, 
            lR[i] + round(dR[i])))

def f_theta():
    return ([0.8 for _ in range(nH)], 
            [random.gauss(0, 0.2) for _ in range(nH)])

def hF():
    return xR[0] - xH[0] - L - s_min
def hF_dot():
    return vR[0] - vH[0]
def hF_ddot():
    return compute_u()[0][0] - f_theta()[0][0]
def get_alphabeta():
    p1 = max(0.1, -hF_dot() / hF())
    p2 = max(0.1, -(hF_ddot() + p1 * hF_dot()) 
                    / (hF_dot() + p1 * hF()))
    return (p1 + p2, p1 * p2)

def hL():
    return vH_max[0] - vR[0] - dvH_th[0]
def hL_dot():
    return vR[0] - vH[0]
def get_p3():
    return max(0.1, -hL_dot() / hL())

def compute_u():
    phiF = f_theta()[0][0] - alpha * hF_dot() - beta * hF()
    uF = max(0, phiF)
    uL = min(0, p3 * hL())
    u_star = uF if uF**2 < uL**2 else uL
    return ([u_star for _ in range(nR)], 
            [0 for _ in range(nR)])

if __name__ == '__main__':

    for i in range(nH):
        xH.append(-3*L)
        vH.append(25)
        lH.append(0)
    for i in range(nR):
        xR.append(0)
        vR.append(25)
        lR.append(0)

    alpha, beta = get_alphabeta()
    p3 = get_p3()

    for t in range(T):
        u = compute_u()
        apply_control(u)
        plot(xH[0])