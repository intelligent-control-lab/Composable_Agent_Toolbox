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
dvH_th = [5]

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

def hF(iH, iR):
    return xR[iR] - xH[iH] - L - s_min
def hF_dot(iH, iR):
    return vR[iR] - vH[iH]
def hL(iH, iR):
    return vH_max[iH] - vR[iR] - dvH_th[iH]
def hL_dot(iH, iR):
    return vR[iR] - vH[iH]

def compute_u():
    return ([1 for _ in range(nR)], 
            [0 for _ in range(nR)])

if __name__ == '__main__':

    for i in range(nH):
        xH.append(-2*L)
        vH.append(25)
        lH.append(0)
    for i in range(nR):
        xR.append(0)
        vR.append(25)
        lR.append(0)

    for t in range(T):
        apply_control(compute_u())
        plot(xR[0])