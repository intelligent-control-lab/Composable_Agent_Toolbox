import math
from queue import Queue
import time

import numpy as np
from scipy.spatial import KDTree

import matlab
import matlab.engine

import matplotlib.pyplot as plt

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
        # print(np.linalg.norm(s1 - s2))
        uv = (s1 - s2) / (np.linalg.norm(s1 - s2) + 0.001) # TODO: why is norm ever 0?
        # print(s1, s2, mag, uv)
        return mag * uv

    def _intersects(self, s1, s2, eps=0):
        return np.linalg.norm(s1 - s2) <= self.r + self.r + eps

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
            s1, s2 = self.spheres[s_i], self.spheres[s_j]
            v1 = self._compute_dv(s1, s2)
            v2 = self._compute_dv(s2, s1)
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
            p_i = self.s2p[s_i]
            s_i_path = int(np.where(self.paths[p_i] == self.spheres[s_i])[0][0]) # TODO: make more efficient
            log.append((self.s2p[s_i], s_i_path))

            if s_i_path == 0:
                v1 = np.zeros(3) # locked in place if first sphere in path
            if s_i_path == len(self.paths[p_i]) and self.at_goal[p_i]:
                v1 = v1[2] * np.array([0, 0, 1]) # only move w.r.t. time if at goal

            for s_j in self._sweep(s_i, v1):
                if s_i == s_j: # prevent infinite loop
                    continue
                if self.s2p[s_i] == self.s2p[s_j]:
                    continue
                s_trans = self.spheres[s_i] + v1
                v2 = self._compute_dv(self.spheres[s_j], s_trans)
                q.put((s_j, v2))

            # apply path shift and check for further intersections
            self._path_shift(p_i, s_i_path, v1)
            for i, s3 in enumerate(self.paths[p_i]):
                query = self.tree.query_ball_point(s3, self.r + self.r)
                for inter_i in query:
                    if self.s2p[inter_i] == p_i:
                        continue
                    v3 = self._compute_dv(s3, self.spheres[inter_i])
                    v_inter = self._compute_dv(self.spheres[inter_i], s3)
                    s_i = int(np.where(self.spheres == s3)[0][0]) # TODO: make more efficient
                    q.put((s_i, v3)) 
                    q.put((inter_i, v_inter))

        return S, V, log

    def _path_shift(self, p_i, s_i, dv):

        s0 = self.paths[p_i][s_i]
        d_max = max([np.linalg.norm(s - s0) for s in self.paths[p_i]])
        mu0 = np.linalg.norm(dv)
        uv = dv / (np.linalg.norm(dv) + 0.001) # TODO: shouldn't have to do this

        # apply path shift
        for i, s in enumerate(self.paths[p_i]):
            if i == 0:
                continue
            d = np.linalg.norm(s0 - s)
            if i == s_i:
                mu = mu0
            else:
                mu = mu0 * math.exp(-self.gamma[p_i] * (d / d_max)**2)
            vec = mu * uv
            if self.at_goal[p_i] and i == len(self.paths[p_i]):
                vec = vec[2] * np.array([0, 0, 1]) # only t component
            self.paths[p_i][i] += vec

        # resolve time inversions
        for i in range(1, len(self.paths[p_i])):
            s_cur = self.paths[p_i][i]
            s_last = self.paths[p_i][i - 1]
            self.paths[p_i][i][2] = min(s_cur[2], s_last[2] + self._deltat(i - 1, i, p_i))
        
    def _deltat(self, i, j, p_i):
        return 0.02
        # s1 = self.paths[p_i][i]
        # s2 = self.paths[p_i][j]
        # d = np.linalg.norm(s2 - s1)
        # v = (self.vel[p_i][i].T @ (s2 - s1)) / d # component of velocity in direction of path
        # a = self.a_max[p_i]
        # return (-v + math.sqrt(v**2 + 2 * a * d)) / a

    def _optimize(self, S, V, P, pri, rad):
        print("Optimizing...")
        n = S.shape[0]
        S = matlab.double(S.tolist())
        V = matlab.double(V.tolist())
        P = matlab.double(P.tolist())
        pri = matlab.double(pri.tolist())
        rad = matlab.double(rad.tolist())
        X, w, fval = self.eng.optimize(S, V, P, pri, rad, n, nargout=3)
        return np.array(X)

    def _pseudo_connect(self, p_i, end=False, clean=True):
        
        if end:
            # insert pseudo-waypoints between p[-2] and p[-1]
            p = self.paths[p_i]
            num = math.ceil(np.linalg.norm(p[-1] - p[-2]) / (self.r + self.r)) - 1
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
                self.vel[p_i].insert(-1, v_last + v_end / num)
                v_last = self.vel[p_i][-1]

        else:
            # insert pseudo-waypoints anywhere necessary
            p = self.paths[p_i]
            p_new = []
            vel_new = []
            is_pseudo = []
            for i in range(len(p)):
                p_new.append(p[i])
                vel_new.append(self.vel[p_i][i])
                is_pseudo.append(self.pseudo[p_i][i])
                if i == len(p) - 1:
                    continue
                num = math.ceil(np.linalg.norm(p[i + 1] - p[i]) / (self.r + self.r)) - 1
                s1 = p[i]
                s2 = p[i + 1]
                v1 = self.vel[p_i][i]
                v2 = self.vel[p_i][i + 1]
                for j in range(num):
                    mag = self.r + self.r
                    uv = (s2 - s1) / np.linalg.norm(s2 - s1)
                    pseu = s1 + mag * uv
                    p_new.append(pseu)
                    is_pseudo.append(True)
                    self.spheres.append(pseu)
                    self.s2p.append(p_i)
                    vel_new.append(v1 + v2 / num)
                    s1 = pseu
                    v1 = vel_new[-1]
            self.paths[p_i] = p_new
            self.vel[p_i] = vel_new
            self.pseudo[p_i] = is_pseudo

        # TODO: remove unnecessary pseudo-waypoints:
        # iterate thru path, for each p[i] find the 
        # farthest p[j] that intersects it where j > i,
        # then delete everything between i and j
        if clean:
            print(f"Cleaning {p_i}")
            p = self.paths[p_i]
            print(f"BEFORE CLEAN:\n{p}\n")
            p_new = []
            vel_new = []
            is_pseudo = []
            i = 0
            while i < len(p):
                print(i, p[i], len(p))
                p_new.append(p[i])
                vel_new.append(self.vel[p_i][i])
                is_pseudo.append(self.pseudo[p_i][i])
                if i == len(p) - 1:
                    break
                j = i + 1
                while j < len(p) and self._intersects(p[i], p[j], eps=1e-5): # j is first non-intersecting sphere, or len(p) if none
                    j += 1
                for k in range(i + 1, j - 1):
                    s_i = np.where(self.spheres == p[k])[0][0] # TODO: make more efficient
                    self.spheres.pop(s_i)
                    self.s2p.pop(s_i)
                i = j - 1
            self.paths[p_i] = p_new
            self.vel[p_i] = vel_new
            self.pseudo[p_i] = is_pseudo
            print("done cleaning")
            print(f"AFTER CLEAN:\n{self.paths[p_i]}\n")

    
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

        self._pseudo_connect(p_i, end=True, clean=False)

        self.tree = KDTree(self.spheres) # reinitialize tree

    def get_path(self, p_i: int) -> list[np.array]:

        path = self.paths[p_i]
        # print(f"ORIGINAL:\n{path}\n")
        authentic = [s for s_i, s in enumerate(path) if not self.pseudo[p_i][s_i]] # remove pseudo-waypoints
        pos_vel_t = [np.append(np.concatenate((s[:2], v), axis=None), s[2]) # [x1, x2, v1, v2, t]
            for s, v in zip(authentic, self.vel[p_i])] 
        
        # print(f"POSVELT: {pos_vel_t}")

        # normalize dt between waypoints using interpolation
        normalized = []
        components = [[s[i] for s in pos_vel_t] for i in range(4)]
        t = [s[4] for s in pos_vel_t]
        t_i = 0
        t_max = t[-1]
        while t_i <= t_max:
            x1, x2, v1, v2 = [np.interp(t_i, t, c) for c in components]
            normalized.append(np.array([x1, x2, v1, v2]))
            t_i += self.dt[p_i]

        # print(f"NORMALIZED:\n{normalized}\n")
        return normalized

    def resolve(self) -> None:
        S, V, log = self._simulate()
        if len(S) == 0:
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter([s[0] for s in S], [s[1] for s in S], [s[2] for s in S])
        plt.show()

        S, V = np.array(S), np.array(V)
        P = self._get_P(S, log)
        pri = np.array([self.priority[p_i] for p_i, _ in log])
        rad = np.ones(S.shape[0])
        X = self._optimize(S, V, P, pri, rad)
        print("Optimization finished!")
        # for i in range(len(S)):
            # print(f"s: {S[i]} ### x: {X[i]} ### p_i: {log[i][0]} ### s_i: {log[i][1]}")

        for (p_i, s_i), x in zip(log, X):
            self._path_shift(p_i, s_i, x - self.paths[p_i][s_i])
        
        for p_i in range(len(self.paths)):
            self._pseudo_connect(p_i)

        for p in self.paths:
            print(p[0], p[1])
            if not self._intersects(p[0], p[1]):
                print("NOT CONNECTED!!!!!!!!!!")

    def _get_P(self, S, log):
        # P is a n*2 matrix containing pairs of spheres in S (1-indexed)
        # that don't belong to the same path
        P = []
        n = S.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if log[i][0] == log[j][0]: # belong to same path
                    continue
                P.append((i + 1, j + 1))
        return np.asarray(P)
