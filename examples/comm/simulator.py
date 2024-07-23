import math
import random
import matplotlib.pyplot as plt
import numpy as np
import trajectory_predictor

class Simulator:
    def __init__(self, x_init, m, n, dt):
        self.x = x_init
        self.m = m
        self.n = n
        self.dt = dt
        self.t = 0
        self.fig, self.ax = plt.subplots()
        self.done = [-1 for _ in range(n)]

    def add_infra(self, infra):
        self.infra = infra

    def vis(self):
        self.ax.cla()
        self.ax.axis([-100, 100, -100, 100])
        for i in range(self.m):
            self.ax.scatter(self.x['pH'][i][0], self.x['pH'][i][1], c='b')
            self.ax.scatter(self.x['gH'][i][0], self.x['gH'][i][1], c='black')
            self.ax.scatter([p[0] for p in self.x['path'][i].values()], [p[1] for p in self.x['path'][i].values()], s=2)
        for i in range(self.n):
            self.ax.scatter(self.x['pR'][i][0], self.x['pR'][i][1], c='r')
            self.ax.scatter(self.x['gR'][i][0], self.x['gR'][i][1], c='grey')
        for p1, p2 in self.x['walls']:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo', linestyle="--", markersize=0)
            
        plt.pause(0.001)

    def f(self):
        aH = []
        self.t = round(self.t, 1)
        for i in range(self.m):
            p_ref = np.array([0, 0])
            tangent = np.array([0, 0])
            if round(self.t + self.dt, 1) in self.x['path'][i]:
                p_ref = self.x['path'][i][self.t]
                # tangent = self.x['path'][i][self.t + self.dt] - self.x['path'][i][self.t]
                tangent = self.x['path'][i][round(self.t + self.dt, 1)] - self.x['pH'][i]
            else:
                p_ref = self.x['gH'][i]
                tangent = self.x['gH'][i] - self.x['pH'][i]
            alpha = 1 # TODO: tune
            v_ref = alpha * tangent / np.linalg.norm(tangent)
            kp = 10
            kv = 2
            aH.append(kp * (p_ref - self.x['pH'][i]) + kv * (v_ref - self.x['vH'][i]))

        return aH

    def u(self): # TODO: implement PD control here
        aR = []
        for i in range(self.n):
            if self.done[i] != -1: # already at goal
                aR.append(np.array([0., 0.]))
            elif np.linalg.norm(self.x['gR'][i] - self.x['pR'][i]) < 5: # just reached goal
                self.x['vR'][i] = np.array([0., 0.])
                aR.append(np.array([0., 0.]))
                self.done[i] = self.t

            else: # move to goal, slow for obstacles
                dir = self.x['gR'][i] - self.x['pR'][i]
                speed2 = self.x['vR'][i].T @ self.x['vR'][i]
                a_max = 10 # TODO: tune
                dist2 = trajectory_predictor.collision_dist2(self.x, self.m, self.infra, i)
                print(round((speed2 / (2 * a_max))**2, 3))
                sig = -1 if dist2 < round((speed2 / (2 * a_max))**2, 3) else 1 # brake if needed, else accelerate
                print(sig)
                aR.append(dir / np.linalg.norm(dir) * a_max * sig)

        return aR

    def move(self):

        aH = self.f()
        aR = self.u()

        v_max = 10

        for i in range(self.m):
            self.x['vH'][i] += aH[i] * self.dt
            # self.x['vH'][i] = min(v_max, self.x['vH'][i])
            self.x['pH'][i] += self.x['vH'][i] * self.dt
            
        for i in range(self.n):
            self.x['vR'][i] += aR[i] * self.dt
            # self.x['vR'][i] = min(v_max, self.x['vR'][i])
            self.x['pR'][i] += self.x['vR'][i] * self.dt
            
        self.x['aH'] = aH
        self.x['aR'] = aR

        self.t += self.dt

        return self.x
