import math
from queue import Queue
import time

import numpy as np
from scipy.spatial import KDTree

import matlab
import matlab.engine

import matplotlib.pyplot as plt

class SpaceTimeGrid:

    def __init__(self, 
        paths: list[list[np.array]], r: float, dt: list[float],
        a_max: list[float], gamma: list[float], priority: list[float], 
        obs_paths: list[list[np.array]]=[], obs_dt: list[float]=[]
    ) -> None:
        self.paths = [
            [np.append(s, dt[p_i] * i) for i, s in enumerate(p)] # add time dimension to all waypoints
        for p_i, p in enumerate(paths)] 
        self.r = r # radius of waypoints on paths
        self.dt = dt # dt of each agent planner
        self.a_max = a_max # max acceleration of agent paths
        self.gamma = gamma # flexibility constant of agent paths
        self.priority = priority # priority of agent paths
        self.obs_paths = [
            [np.append(s, obs_dt[p_i] * i) for i, s in enumerate(p)] # add time dimension to all waypoints
        for p_i, p in enumerate(obs_paths)] 
        self.obs_dt = obs_dt # dt of each obs planner
        self.vel = [[np.array([0, 0]) for _ in p] for p in self.paths] # velocity at each waypoint for agent paths
        self.at_goal = [False for i in range(len(self.paths))] # whether agent has reached goal
        self.tree = KDTree([s for p in self.paths for s in p])
        self.obs_tree = KDTree([s for p in self.obs_paths for s in p])
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r"C:\Users\aniru\OneDrive\Documents\Code\ICL\OptimizationSolver", nargout=0)

        # performance
        self.opt_num = 0
        self.opt_time = 0
        self.tree_time = 0

        print("STG initialized...")

    def _compute_dv(self, s1, s2):
        mag = self.r + self.r - np.linalg.norm(s1 - s2)
        uv = (s1 - s2) / np.linalg.norm(s1 - s2)
        return mag * uv

    def _intersects(self, s1, s2, eps=0):
        return np.linalg.norm(s1 - s2) <= self.r + self.r + eps

    def _tangent(self, s1, s2):
        return np.linalg.norm(s1 - s2) == self.r + self.r

    def _get_index(self, arr, s):
        for i in range(len(arr)):
            if np.allclose(s, arr[i]):
                return i

    def _idx2sp(self, paths, idx):
        # convert index value receieved by self.tree to p_i, s_i
        map = [(p_i, s_i)
            for p_i in range(len(paths)) 
                for s_i in range(len(paths[p_i]))]
        return map[idx][0], map[idx][1]

    def _sweep(self, p_i, s_i, v):

        s = self.paths[p_i][s_i]
        inter = []
        inter_obs = []
        s_trans = s.copy()

        # translate s_trans along v
        while np.linalg.norm(s_trans - s) < np.linalg.norm(v):
            query = self.tree.query_ball_point(s_trans, self.r + self.r)
            inter += [tuple(self._idx2sp(self.paths, idx)) for idx in query]
            query_obs = self.obs_tree.query_ball_point(s_trans, self.r + self.r)
            inter_obs += [tuple(self._idx2sp(self.obs_paths, idx)) for idx in query_obs]
            s_trans = s_trans + self.r * v / np.linalg.norm(v) # iterate by distance r

        s_trans = s + v # iterate to end of vector
        query = self.tree.query_ball_point(s_trans, self.r + self.r)
        inter += [tuple(self._idx2sp(self.paths, idx)) for idx in query]
        query_obs = self.obs_tree.query_ball_point(s_trans, self.r + self.r)
        inter_obs += [tuple(self._idx2sp(self.obs_paths, idx)) for idx in query_obs]
        
        return inter, inter_obs

    def _simulate(self):

        S = [] # outstanding spheres
        V = [] # DVs of S
        log = [] # list of tuples (index of S[i] in its path, p_i)
        
        q = Queue()
        vis = [[False for _ in p] for p in self.paths]
        vis_obs = [[False for _ in p] for p in self.obs_paths]

        # initially intersecting
        query = self.tree.query_pairs(self.r + self.r)
        query_spheres = []
        for idx1, idx2 in query:
            p_i, s_i = self._idx2sp(self.paths, idx1)
            p_j, s_j = self._idx2sp(self.paths, idx2)
            if p_i == p_j:
                continue
            s1 = self.paths[p_i][s_i]
            s2 = self.paths[p_j][s_j]
            query_spheres.append((s1, s2, np.linalg.norm(s1 - s2)))
        query_spheres.sort(key=lambda tup: tup[2])

        for s1, s2, _ in query_spheres:
            v1 = self._compute_dv(s1, s2)
            v2 = self._compute_dv(s2, s1)
            q.put((p_i, s_i, v1, False))
            q.put((p_j, s_j, v2, False))

        # initially intersecting w/ obs
        query_obs = self.tree.query_ball_tree(self.obs_tree, self.r + self.r)
        query_obs_spheres = []
        for i in range(len(query_obs)):
            for j in query_obs[i]:
                p_i, s_i = self._idx2sp(self.paths, i)
                p_j, s_j = self._idx2sp(self.obs_paths, j)
                s_ag = self.paths[p_i][s_i]
                s_ob = self.obs_paths[p_j][s_j]
                query_obs_spheres.append((s_ag, s_ob, np.linalg.norm(s_ag - s_ob)))
        query_obs_spheres.sort(key=lambda tup: tup[2])

        for s_ag, s_ob, _ in query_obs_spheres:
            v1 = self._compute_dv(s_ag, s_ob)
            v2 = np.zeros(3)
            q.put((p_i, s_i, v1, False))
            q.put((p_j, s_j, v2, True))

        # bfs sim
        while not q.empty():

            p_i, s_i, v1, is_ob = q.get()
            
            if is_ob:
                if vis_obs[p_i][s_i]: # prevent infinite loop
                    continue
                vis_obs[p_i][s_i] = True
            else:
                if vis[p_i][s_i]: # prevent infinite loop
                    continue
                vis[p_i][s_i] = True
            
            if not is_ob:
                if s_i == 0:
                    v1 = np.zeros(3) # locked in place
                if self.at_goal[p_i] and s_i == len(self.paths[p_i]) - 1:
                    v1 = v1[2] * np.array([0, 0, 1]) # only move w.r.t. time if at goal

            if is_ob:
                S.append(self.obs_paths[p_i][s_i])
            else:
                S.append(self.paths[p_i][s_i])
            V.append(v1)
            log.append((p_i, s_i, is_ob))

            if np.all(v1 == 0): # anything after this is guaranteed not to be an obs sphere
                continue

            sw, sw_obs = self._sweep(p_i, s_i, v1)
            for p_j, s_j in sw:
                if p_i == p_j:
                    continue
                s_trans = self.paths[p_i][s_i] + v1
                v2 = self._compute_dv(self.paths[p_j][s_j], s_trans)
                q.put((p_j, s_j, v2, False))
            for p_j, s_j in sw_obs:
                v2 = np.zeros(3)
                q.put((p_j, s_j, v2, True))

            # TODO: when a path shift is applied, does every sphere in the path then need to be added to S?

            # simulate path shift and check for further intersections
            # add a path-shifted sphere only if it has a conflict
            sim_shift, sim_vel = self._path_shift(p_i, s_i, v1)
            for i, s3 in enumerate(sim_shift):
                if i == p_i:
                    continue
                query = self.tree.query_ball_point(s3, self.r + self.r)
                inter = [tuple(self._idx2sp(self.paths, idx)) for idx in query]
                conflict = False
                for p_k, _ in inter:
                    if p_i != p_k:
                        conflict = True
                        break
                if conflict:
                    q.put((p_i, i, s3 - self.paths[p_i][i], False))

            # # add all path-shifted spheres
            # sim_shift, sim_vel = self._path_shift(p_i, s_i, v1)
            # for i, s3 in enumerate(sim_shift):
            #     q.put((p_i, i, s3 - self.paths[p_i][i], False))

        return S, V, log

    def _path_shift(self, p_i, s_i, dv):

        if np.all(dv == 0):
            return self.paths[p_i], self.vel[p_i]

        p = np.array(self.paths[p_i]).copy() # deep copy
        s0 = p[s_i]
        d_max = max([np.linalg.norm(s - s0) for s in p])

        # apply path shift
        for i, s in enumerate(p):
            d = np.linalg.norm(s0 - s)
            mu = math.exp(-self.gamma[p_i] * (d / d_max)**2)
            vec = mu * dv
            if self.at_goal[p_i] and i == len(p) - 1:
                vec = vec[2] * np.array([0, 0, 1]) # only t component
            if i > 0: # don't move first sphere
                p[i] += vec

        # resolve time inversions and motion constraints
        vel = [np.zeros(2) for _ in p]
        for i in range(1, len(p)):
            delt = self._deltat(p[i - 1], p[i], vel[i - 1], self.a_max[p_i])
            vel[i] = vel[i - 1] + self.a_max[p_i] * delt
            p[i][2] = p[i - 1][2] + delt

        return p, vel
        
    def _deltat(self, s1, s2, v1, a): # TODO: redo this method
        d = np.linalg.norm(s2 - s1)
        v = (v1.T @ (s2[:2] - s1[:2])) / d # component of velocity in direction of path
        return (-v + math.sqrt(v**2 + 2 * a * d)) / a

    def _optimize(self, S, V, P, pri, rad):
        n = S.shape[0]
        if n == 0:
            return np.array([])
        self.opt_num += 1
        print("Optimizing...")
        S = matlab.double(S.tolist())
        V = matlab.double(V.tolist())
        P = matlab.double(P.tolist())
        pri = matlab.double(pri.tolist())
        rad = matlab.double(rad.tolist())
        start = time.time()
        X, w, fval = self.eng.optimize(S, V, P, pri, rad, n, nargout=3)
        end = time.time()
        self.opt_time += end - start
        print("Optimization finished!")
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
                self.vel[p_i].insert(-1, v_last + v_end / num)
                v_last = self.vel[p_i][-1]

        else:
            # insert pseudo-waypoints anywhere necessary
            p = self.paths[p_i]
            p_new = []
            vel_new = []
            for i in range(len(p)):
                p_new.append(p[i])
                vel_new.append(self.vel[p_i][i])
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
                    vel_new.append(v1 + v2 / num)
                    s1 = pseu
                    v1 = vel_new[-1]
            self.paths[p_i] = p_new
            self.vel[p_i] = vel_new

        # Remove unnecessary pseudo-waypoints:
        # iterate thru path, for each p[i] find the 
        # first p[j] that doesn't intersect p[i] where 
        # j > i, then delete everything between i and j - 1
        if clean:
            p = self.paths[p_i]
            p_new = []
            vel_new = []
            i = 0
            while i < len(p):
                p_new.append(p[i])
                vel_new.append(self.vel[p_i][i])
                if i == len(p) - 1:
                    break
                j = i + 1
                while j < len(p) and self._intersects(p[i], p[j], eps=1e-5):
                    j += 1
                i = j - 1
            self.paths[p_i] = p_new
            self.vel[p_i] = vel_new

    def _pseudo_connect_obs(self, p_i):

        p = self.obs_paths[p_i]

        num = math.ceil(np.linalg.norm(p[-1] - p[-2]) / (self.r + self.r)) - 1
        for i in range(num):
            # insert tangent to second to last sphere in direction towards last sphere
            mag = self.r + self.r
            uv = (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2])
            pseu = p[-2] + mag * uv
            self.obs_paths[p_i].insert(-1, pseu)

        p_new = []
        i = 0
        while i < len(p):
            p_new.append(p[i])
            if i == len(p) - 1:
                break
            j = i + 1
            while j < len(p) and self._intersects(p[i], p[j], eps=1e-5):
                j += 1
            i = j - 1
        self.obs_paths[p_i] = p_new
    
    def set_at_goal(self, p_i: int, val: bool) -> None:
        self.at_goal[p_i] = val

    def update_path(self, p_i: int, s: np.array, v: np.array) -> None:

        t = self.paths[p_i][-1][2] + self.dt[p_i]
        s_new = np.append(s, t)

        self.paths[p_i].append(s_new)
        self.vel[p_i].append(v)

        self._pseudo_connect(p_i, end=True, clean=False)

        start = time.time()
        self.tree = KDTree([s for p in self.paths for s in p]) # reinitialize tree
        end = time.time()
        self.tree_time += end - start

    def update_obs_path(self, p_i: int, s: np.array) -> None:

        t = self.obs_paths[p_i][-1][2] + self.obs_dt[p_i]
        s_new = np.append(s, t)

        self.obs_paths[p_i].append(s_new)

        self._pseudo_connect_obs(p_i)

        start = time.time()
        self.obs_tree = KDTree([s for p in self.obs_paths for s in p]) # reinitialize tree
        end = time.time()
        self.tree_time += end - start

    def get_path(self, p_i: int) -> list[np.array]:

        pos_vel_t = [np.append(np.concatenate((s[:2], v), axis=None), s[2]) # [x1, x2, v1, v2, t]
            for s, v in zip(self.paths[p_i], self.vel[p_i])] 
        
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

        return normalized

    def resolve(self) -> None:

        S, V, log = self._simulate()

        for i in range(len(S)):
            print(f"S[i]: {S[i]} ##### V[i]: {V[i]} ##### log[i]: {log[i]}")

        S, V = np.array(S), np.array(V)
        P = self._get_P(S, log)
        pri = np.array([self.priority[p_i] if is_ob else 1 for p_i, _, is_ob in log])
        rad = np.array([self.r for i in range(S.shape[0])])
        X = self._optimize(S, V, P, pri, rad)

        # # plot paths and outstanding spheres
        # if not S.shape[0] == 0:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     c = ["blue", "red", "green"]
        #     for i, p in enumerate(self.paths + self.obs_paths):
        #         ax.scatter([s[0] for s in p], [s[1] for s in p], [s[2] for s in p], color=c[i])
        #     ax.scatter([s[0] for s in S], [s[1] for s in S], [s[2] for s in S], color="yellow", s=100)
        #     plt.show()

        for (p_i, s_i, is_ob), x in zip(log, X):
            if is_ob:
                continue
            vec = x - self.paths[p_i][s_i]
            self.paths[p_i], self.vel[p_i] = self._path_shift(p_i, s_i, vec)

        for p_i in range(len(self.paths)):
            self._pseudo_connect(p_i)

        start = time.time()
        self.tree = KDTree([s for p in self.paths for s in p]) # reinitialize tree
        end = time.time()
        self.tree_time += end - start

    def _get_P(self, S, log):
        # P is a n*2 matrix containing pairs of spheres in S (1-indexed)
        # that don't belong to the same path
        P = []
        n = S.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if log[i][0] == log[j][0] and not log[i][2] and not log[j][2]: # belong to same agent path
                    continue
                if log[i][2] and log[j][2]: # both obstacle paths
                    continue
                P.append((i + 1, j + 1))
        return np.asarray(P)
