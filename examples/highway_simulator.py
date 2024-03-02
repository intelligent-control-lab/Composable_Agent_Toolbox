import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class HighwaySimulator:
    def __init__(self, x_init, m, n, q, L, idmH, idmR, dt):
        self.x = x_init
        self.m = m
        self.n = n
        self.q = q
        self.L = L
        self.idmH = idmH
        self.idmR = idmR
        self.dt = dt

        self.fig, self.ax = plt.subplots()

    def vis(self, t, pov):
        
        self.ax.cla()
        self.ax.axis([-4*self.L, 4*self.L, pov - 4*self.L, pov + 4*self.L])
        self.ax.axvline(x=-self.L/2, color="black", linestyle="--")
        self.ax.axvline(x=self.L/2, color="black", linestyle="--")
        self.ax.axvline(x=-3*self.L/2, color="black", linestyle="-")
        self.ax.axvline(x=3*self.L/2, color="black", linestyle="-")

        self.ax.text(-4*self.L, pov + 4*self.L, "t = " + str(round(t, 1)))
        for i in range(self.m):
            self.ax.text(self.x['lH'][i]*self.L, self.x['pH'][i], str(round(self.x['vH'][i], 1)))
        for i in range(self.n):
            self.ax.text(self.x['lR'][i]*self.L, self.x['pR'][i], str(round(self.x['vR'][i], 1)))
        for i in range(self.q):
            self.ax.text(self.x['lB'][i]*self.L, self.x['pB'][i], str(round(self.x['vB'][i], 1)))

        for i in range(self.m):
            self.ax.add_patch(Rectangle(
                (self.x['lH'][i]*self.L - self.L/4, self.x['pH'][i] - self.L/2), 
                self.L/2, self.L, 
                facecolor='blue'))
        for i in range(self.n):
            self.ax.add_patch(Rectangle(
                (self.x['lR'][i]*self.L - self.L/4, self.x['pR'][i] - self.L/2), 
                self.L/2, self.L, 
                facecolor='red'))
        for i in range(self.q):
            self.ax.add_patch(Rectangle(
                (self.x['lB'][i]*self.L - self.L/4, self.x['pB'][i] - self.L/2), 
                self.L/2, self.L, 
                facecolor='green'))

        plt.pause(0.001)

    def get_following(self, i, l):
        fol = -1
        for j in range(self.m):
            if j == i or self.x['lH'][j] != l or self.x['pH'][i] > self.x['pH'][j]:
                continue
            if fol == -1 or self.x['pH'][j] < self.x['pH'][fol]:
                fol = j
        if fol != -1:
            return self.x['pH'][fol], self.x['vH'][fol]
        for j in range(self.n):
            if self.x['lR'][j] != l or self.x['pH'][i] > self.x['pR'][j]:
                continue
            if fol == -1 or self.x['pR'][j] < self.x['pR'][fol]:
                fol = j
        if fol != -1:
            return self.x['pR'][fol], self.x['vR'][fol]
        for j in range(self.q):
            if self.x['lB'][j] != l or self.x['pH'][i] > self.x['pB'][j]:
                continue
            if fol == -1 or self.x['pB'][j] < self.x['pB'][fol]:
                fol = j
        if fol != -1:
            return self.x['pB'][fol], self.x['vB'][fol]
        return None

    def is_occupied(self, p, l):
        for i in range(self.m):
            if self.x['lH'][i] == l and abs(p - self.x['pH'][i]) < self.L + self.L:
                return True
        for j in range(self.n):
            if self.x['lR'][j] == l and abs(p - self.x['pR'][j]) < self.L + self.L:
                return True
        for k in range(self.q):
            if self.x['lB'][k] == l and abs(p - self.x['pB'][k]) < self.L + self.L:
                return True
        return False

    def f(self):
        aH = []
        dH = []
        for i in range(self.m):
            fol = self.get_following(i, self.x['lH'][i])
            aH.append(self.idmH[i].free_road(self.x['vH'][i]) if fol is None 
                    else self.idmH[i].idm(self.x['pH'][i], fol[0], self.x['vH'][i], fol[1]))
            # incentive = False
            # if fol is not None:
            #     fol_new = get_following(i, -1)
            #     vel_new = v0 if fol_new is None else fol_new[1]
            #     incentive = vel_new - fol[1] >= vH_th
            safety = not self.is_occupied(self.x['pH'][i], self.x['lH'][i] - 1)
            # dH.append(0 if fol is None 
            #           else -(incentive and safety))
            dH.append(0 if fol is None else -safety)
        return (aH, dH)

    def u(self):
        aR = []
        dR = [0 for _ in range(self.n)]
        for i in range(self.n):
            fol = self.get_following(i, self.x['lR'][i])
            aR.append(self.idmR[i].free_road(self.x['vR'][i]) if fol is None 
                    else self.idmR[i].idm(self.x['pR'][i], fol[0], self.x['vR'][i], fol[1]))
        return (aR, dR)

    def move(self, use_idm=True):

        aH, dH = self.f() if use_idm else ([0 for _ in range(self.m)], 
                                [0 for _ in range(self.m)])
        aR, dR = self.u() if use_idm else ([0 for _ in range(self.n)], 
                                [0 for _ in range(self.n)])

        for i in range(self.m):
            # aH[i] = min(a_max, max(-b_max, aH[i]))
            self.x['vH'][i] += aH[i] * self.dt
            self.x['pH'][i] += self.x['vH'][i] * self.dt
            self.x['lH'][i] = max(-1, min(1, 
                self.x['lH'][i] + round(dH[i])))
            
        for i in range(self.n):
            # aR[i] = min(a_max, max(-b_max, aR[i]))
            self.x['vR'][i] += aR[i] * self.dt
            self.x['pR'][i] += self.x['vR'][i] * self.dt
            self.x['lR'][i] = max(-1, min(1, 
                self.x['lR'][i] + round(dR[i])))
            
        for i in range(self.q):
            self.x['pB'][i] += self.x['vB'][i] * self.dt
            
        self.x['aH'] = aH
        self.x['aR'] = aR

    def sense(self, observer, subject):
        pos = self.x['pH'][subject]
        vel = self.x['vH'][subject]
        dist = abs(pos - self.x['pR'][observer])
        alpha = (0.01, 0.01) # TODO: tune params
        pos += random.gauss(0, alpha[0]*dist**2 + alpha[1]*pos)
        vel += random.gauss(0, alpha[0]*dist**2 + alpha[1]*vel)
        return (pos, vel)
