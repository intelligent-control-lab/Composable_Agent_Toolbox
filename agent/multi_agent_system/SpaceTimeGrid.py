import math
from queue import Queue
import time

import numpy as np
from scipy.spatial import KDTree

import matlab
import matlab.engine

class SpaceTimeGrid:

    def __init__(self, paths: list[list[np.array]], a_max: list[float], gamma: list[float], priority: list[float], dt: list[float]) -> None:
        self.paths = [
            [np.append(s, dt[p_i] * i) for i, s in enumerate(p)] # add time dimension to all waypoints
        for p_i, p in enumerate(paths)] 
        self.spheres = [s for p in self.paths for s in p]
        self.s2p = [p_i for p_i, p in enumerate(self.paths) for _ in p] # get path index from sphere index
        self.pseudo = [[False for _ in p] for p in self.paths] # whether path[p_i][s_i] is a pseudo-waypoint
        self.n_agents = len(self.paths)
        self.a_max = a_max
        self.gamma = gamma
        self.priority = priority # priority value of each path
        self.dt = dt
        self.tree = KDTree(self.spheres)
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r"C:\Users\aniru\OneDrive\Documents\Code\ICL\OptimizationSolver", nargout=0)
        self.r = 1
        self.at_goal = [False for i in range(len(self.paths))] # whether path has reached goal
        self.vel = [[np.array([0, 0]) for _ in p] for p in self.paths] # velocity at each waypoint for each path

    def _compute_dv(self, s1, s2):
        mag = self.r + self.r - np.linalg.norm(s1 - s2)
        uv = (s1 - s2) / np.linalg.norm(s1 - s2)
        return mag * uv

    def _intersects(self, s1, s2, epsilon=0):
        return np.linalg.norm(s1 - s2) < self.r + self.r - epsilon

    def _tangent(self, s1, s2):
        return np.linalg.norm(s1 - s2) == self.r + self.r
    
    def _sweep(self, s_i, v):

        s = self.spheres[s_i]
        inter = []
        s_trans = s.copy()

        # translate s_trans along v
        while np.linalg.norm(s_trans - s) < np.linalg.norm(v):
            query = self.tree.query_ball_point(s_trans, self.r + self.r)
            inter += query
            s_trans = s_trans + self.r * v / np.linalg.norm(v) # iterate by distance r

        s_trans = s + v # iterate to end of vector
        query = self.tree.query_ball_point(s_trans, self.r + self.r)
        inter += query
        return inter

    def _simulate(self):

        S = [] # outstanding spheres
        V = [] # DVs of S
        log = [] # list of tuples (index of S[i] in its path, p_i)
        
        q = Queue()
        vis = [False for i in range(len(self.spheres))]

        # initially intersecting
        for s_i, s_j in self.tree.query_pairs(self.r + self.r):
            if self.s2p[s_i] == self.s2p[s_j]:
                continue
            v1 = self._compute_dv(s_i, s_j)
            v2 = self._compute_dv(s_j, s_i)
            q.put((s_i, v1))
            q.put((s_j, v2))

        # bfs sim
        while not q.empty():

            s_i, v1 = q.get()
            if vis[s_i]: # prevent infinite loop
                continue
            vis[s_i] = True
            S.append(self.spheres[s_i])
            V.append(v1)
            log.append((self.s2p[s_i], s_i))

            for s_j in self._sweep(s_i, v1):
                if s_i == s_j: # prevent infinite loop
                    continue
                if self.s2p[s_i] == self.s2p[s_j]:
                    continue
                s_trans = self.spheres[s_i] + v1
                v2 = self._compute_dv(self.spheres[s_j], s_trans)
                q.put((s_j, v2))

            # apply path shift and check for further intersections
            p_i = self.s2p[s_i]
            self._path_shift(p_i, s_i, v1)
            for s3 in self.paths[p_i]:
                query = self.tree.query_ball_point(s3, self.r + self.r)
                for inter_i in query:
                    v3 = self.compute_dv(s3, self.spheres[inter_i])
                    q.put(self.spheres.index(s3), v3) # TODO: make more efficient later

        return S, V, log

    def _path_shift(self, p_i, s0, dv):

        d_max = max([np.linalg.norm(s - s0) for s in self.paths[p_i]])
        mu0 = np.linalg.norm(dv)
        uv = dv / np.linalg.norm(dv)

        # apply path shift
        for i, s in enumerate(self.paths[p_i]):
            if i == 0:
                continue
            d = np.linalg.norm(s0 - s)
            mu = mu0 * math.exp(-self.gamma[p_i] * (d / d_max)**2)
            vec = mu * uv
            if self.at_goal[p_i] and i == len(self.paths[p_i]):
                vec = vec[2] * np.array([0, 0, 1]) # only t component
            self.paths[p_i][i] += vec

        # resolve time inversions
        for i in range(len(self.paths[p_i]) - 2, 1, -1):
            s_cur = self.paths[p_i][i]
            s_next = self.paths[p_i][i + 1]
            self.paths[p_i][i][2] = min(s_cur[2], s_next[2] - self._deltat(i, i + 1, p_i))
        
    def _deltat(self, i, j, p_i):
        s1 = self.paths[p_i][i]
        s2 = self.paths[p_i][j]
        d = np.linalg.norm(s2 - s1)
        v = (self.vel[p_i][i].T @ (s2 - s1)) / d # component of velocity in direction of path
        a = self.a_max[p_i]
        return (-v + math.sqrt(v**2 + 2 * a * d)) / a

    def _optimize(self, S, V, rad, pri):
        if S.shape[0] == 0:
            return S
        n = S.shape[0]
        S = matlab.double(S.tolist())
        V = matlab.double(V.tolist())
        rad = matlab.double(rad.tolist())
        pri = matlab.double(pri.tolist())
        X, w, fval = self.eng.optimize(S, V, rad, pri, n, nargout=3)
        return X
    
    def set_at_goal(self, p_i: int, val: bool) -> None:
        self.at_goal[p_i] = val

    def update_path(self, p_i: int, s: np.array, v: np.array) -> None:

        t = self.paths[p_i][-1][2] + self.dt[p_i]
        s_new = np.append(s, t)

        self.paths[p_i].append(s_new)
        self.spheres.append(s_new)
        self.s2p.append(p_i)
        self.pseudo[p_i].append(False)
        self.vel[p_i].append(v)

        # insert pseudo-waypoints
        p = self.paths[p_i]
        num = math.ceil(np.linalg.norm(self.paths[p_i][-1] - self.paths[p_i][-2]) / (self.r + self.r)) - 1
        v_last = self.vel[p_i][-2]
        v_end = self.vel[p_i][-1]
        for i in range(num):
            # insert tangent to second to last sphere in direction towards last sphere
            mag = self.r + self.r
            uv = (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2])
            pseu = p[-2] + mag * uv
            self.paths[p_i].insert(-1, pseu)
            self.pseudo[p_i].insert(-1, True)
            self.spheres.append(pseu)
            self.s2p.append(p_i)
            self.vel[p_i].insert(-1, v_last + (1 / num) * v_end)
            v_last = self.vel[p_i][-1]

        self.tree = KDTree(self.spheres) # reinitialize tree

    def get_path(self, p_i: int) -> list[np.array]:

        path = self.paths[p_i]
        authentic = [s for s_i, s in enumerate(path) if not self.pseudo[p_i][s_i]] # remove pseudo-waypoints
        pos_vel = [np.append(np.concatenate((s[:2], v), axis=None), s[2]) # [x1, x2, v1, v2, t]
            for s, v in zip(authentic, self.vel[p_i])] 
        
        # "normalize" dt between waypoints thru interpolation
        t = 0
        s_i = 0
        normalized = []
        while s_i < len(pos_vel) - 1:
            s1, s2 = pos_vel[s_i], pos_vel[s_i + 1]
            if not (t >= s1[4] and t < s2[4]):
                s_i += 1
                continue
            m = 1 / (s2[4] - s1[4]) * (s2[:4] - s1[:4])
            waypoint = s1[:4] + (t - s1[4]) * m
            normalized.append(waypoint)
            t += self.dt[p_i]

        return normalized

    def resolve(self) -> None:
        S, V, log = self._simulate()
        S, V = np.array(S), np.array(V)
        rad = np.ones(S.shape[0])
        pri = np.array([self.priority[p_i] for _, p_i in log])
        X = self._optimize(S, V, rad, pri)
        for p_i, s_i, x_i in zip(log, X):
            self._path_shift(p_i, s_i, x_i - s_i)
