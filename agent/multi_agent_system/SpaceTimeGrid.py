import math
from queue import Queue
import time

import numpy as np
from scipy.spatial import KDTree

import matlab
import matlab.engine

class SpaceTimeGrid:

    def __init__(self, paths: list[list[np.array]], a_max: list[float], gamma: list[float]) -> None:
        self.paths = paths
        self.spheres = [s for p in paths for s in p]
        self.s2p = [p_i for p_i, p in enumerate(paths) for _ in p] # get path index from sphere index
        self.pseudo = [[False for _ in p] for p in paths] # whether path[p_i][s_i] is a pseudo-waypoint
        self.n_agents = len(paths)
        self.a_max = a_max
        self.gamma = gamma
        self.tree = KDTree(self.spheres)
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r"C:\Users\aniru\OneDrive\Documents\Code\ICL\OptimizationSolver", nargout=0)
        self.r = 1
        self.at_goal = [False for i in range(len(paths))] # whether path has reached goal

        # TODO: maybe integrate this better, perhaps have each waypoint have its own projected velocity??
        self.vel = [np.array([]) for i in range(len(paths))] # current velocity of each agent

    def __init__(self, paths: list[np.array]) -> None:
        self.__init__(paths, np.ones(len(paths)), np.ones(len(paths)))

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
        for i in range(len(self.paths[p_i]) - 1, 1, -1):
            s_cur = self.paths[p_i][i]
            s_next = self.paths[p_i][i + 1]
            self.paths[p_i][i][2] = min(s_cur[2], s_next[2] - self._dt(s_cur, s_next))
        
    def _dt(self, s1, s2, p_i):
        d = np.linalg.norm(s2 - s1)
        v = (self.vel[p_i].T @ (s2 - s1)) / d # component of velocity in direction of path
        a = self.a_max[p_i]
        return (-v + math.sqrt(v**2 + 2 * a * d)) / a

    def _optimize(self, S, V, rad):
        n = S.shape[0]
        S = matlab.double(S.tolist())
        V = matlab.double(V.tolist())
        rad = matlab.double(rad.tolist())
        X, w, fval = self.eng.optimize(S, V, rad, n, nargout=3)
        return X
    
    def set_at_goal(self, p_i: int, val: bool) -> None:
        self.at_goal[p_i] = val

    def update_path(self, p_i: int, s_new: np.array) -> None:

        self.paths[p_i].append(s_new)
        self.spheres.append(s_new)
        self.s2p.append(p_i)
        self.pseudo[p_i].append(False)

        # insert pseudo-waypoints
        p = self.paths[p_i]
        while not self._tangent(p[-1], p[-2]) and not self._intersects(p[-1], p[-2]):
            # insert tangent to second to last sphere in direction towards last sphere
            mag = self.r + self.r
            uv = (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2])
            pseu = p[-2] + mag * uv
            self.paths[p_i].insert(-2, pseu)
            self.spheres.append(pseu)
            self.s2p.append(p_i)
            self.pseudo[p_i].append(True)

        self.tree = KDTree(self.spheres) # reinitialize tree

    def update_vel(self, p_i: int, val: np.array) -> None:
        self.vel[p_i] = val

    def get_path(self, p_i: int) -> list[np.array]:
        path = self.paths[p_i]
        authentic = [s for s_i, s in enumerate(path) if not self.pseudo[p_i][s_i]]
        return authentic

    def resolve(self) -> None:
        S, V, log = self._simulate()
        S, V = np.array(S), np.array(V)
        rad = np.ones(S.shape[0])
        X = self._optimize(S, V, rad)
        for p_i, s_i, x_i in zip(log, X):
            self._path_shift(p_i, s_i, x_i - s_i)
