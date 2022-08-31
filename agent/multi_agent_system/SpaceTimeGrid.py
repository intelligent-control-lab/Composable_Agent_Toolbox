import math
from queue import Queue
import time

import numpy as np
from scipy.spatial import KDTree

import matlab
import matlab.engine

class SpaceTimeGrid:

    def __init__(self, paths, a_max):
        self.paths = paths
        self.n_agents = len(paths)
        self.a_max = a_max
        self.tree = KDTree([s for p in paths for s in p])
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r"C:\Users\aniru\OneDrive\Documents\Code\ICL\OptimizationSolver", nargout=0)
        self.r = 1
        self.at_goal = [False for i in range(len(paths))]
        self.vel = [np.array([]) for i in range(len(paths))]

    def _compute_dv(self, s1, s2):
        mag = self.r + self.r - np.linalg.norm(s1 - s2)
        uv = (s1 - s2) / np.linalg.norm(s1 - s2)
        return mag * uv

    def _intersects(self, s1, s2, epsilon=0):
        return np.linalg.norm(s1 - s2) < self.r + self.r - epsilon

    def _sweep(self, s_cur, v_cur):

        inter = []
        s_trans = s_cur.copy()

        # translate s_trans along v_cur
        while np.linalg.norm(s_trans - s_cur) < np.linalg.norm(v_cur):
            query = self.tree.query_ball_point(s_trans, self.r + self.r)
            for i in query:
                inter.append(self.spheres[i])
            s_trans = s_trans + self.r * v_cur / np.linalg.norm(v_cur) # iterate by distance r

        s_trans = s_cur + v_cur # iterate to end of vector
        query = self.tree.query_ball_point(s_trans, self.r + self.r)
        for i in query:
            inter.append(self.spheres[i])
        return inter

    def _path_shift(self, p_i, s0, dv, gamma=1):

        d_max = max([np.linalg.norm(s - s0) for s in self.paths[p_i]])
        mu0 = np.linalg.norm(dv)
        uv = dv / np.linalg.norm(dv)

        # apply path shift
        for i, s in enumerate(self.paths[p_i]):
            if i == 0:
                continue
            d = np.linalg.norm(s0 - s)
            mu = mu0 * math.exp(-gamma * (d / d_max)**2)
            vec = mu * uv
            if self.at_goal[p_i] and i == len(self.paths[p_i]):
                vec = vec[2] * np.array([0, 0, 1]) # only t component
            self.paths[p_i][i] += vec

        # resolve time inversions
        for i in range(len(self.paths[p_i]) - 1, 1, -1):
            s_cur = self.paths[p_i][i]
            s_next = self.paths[p_i][i + 1]
            self.paths[p_i][i][2] = min(s_cur[2], s_next[2] - self._dt(s_cur, s_next))
        
    def _dt(self, s1, s2, p_i):
        v = np.linalg.norm(self.vel[p_i])
        a = self.a_max[p_i]
        d = np.linalg.norm(s1 - s2)
        return (-v + math.sqrt(v**2 + 2 * a * d)) / a

    def _simulate(self):

        S = [] # outstanding spheres
        V = [] # DVs of S_out
        q = Queue()

        vis = {}
        for s in self.spheres:
            vis[tuple(s)] = False

        # initially intersecting
        for i, j in self.tree.query_pairs(self.r + self.r):
            print(i, j)
            s1, s2 = self.spheres[i], self.spheres[j]
            v1 = self._compute_dv(s1, s2)
            v2 = self._compute_dv(s2, s1)
            q.put((s1, v1))
            q.put((s2, v2))

        # bfs sim
        while not q.empty():
            s_cur, v_cur = q.get()
            if vis[tuple(s_cur)]: # prevent infinite loop
                continue
            vis[tuple(s_cur)] = True
            S.append(s_cur)
            V.append(v_cur)
            for s_new in self._sweep(s_cur, v_cur):
                if (s_new==s_cur).all() or ((s_cur==s1).all() and (s_new==s2).all()) or ((s_cur==s2).all() and (s_new==s1).all()): # prevent infinite loop
                    continue
                s_trans = s_cur + v_cur
                v_new = self._compute_dv(s_new, s_trans)
                q.put((s_new, v_new))

        return S, V

    def _optimize(self, S, V, rad):
        n = S.shape[0]
        S = matlab.double(S.tolist())
        V = matlab.double(V.tolist())
        rad = matlab.double(rad.tolist())
        X, w, fval = self.eng.optimize(S, V, rad, n, nargout=3)
        return X

    def set_at_goal(self, p_i, val):
        self.at_goal[p_i] = val

    # TODO: find incremental querying structure
    def update_path(self, p_i, s):
        pass

    def update_vel(self, p_i, val):
        self.vel[p_i] = val

    def resolve(self):
        S, V = self._simulate()
        S, V = np.array(S), np.array(V)
        rad = np.ones(S.shape[0])
        X = self._optimize(S, V, rad)
        # return dict {agent_i : [path edits]}