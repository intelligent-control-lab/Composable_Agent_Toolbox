import math
from queue import Queue
import time

import numpy as np
from scipy.spatial import KDTree

import matlab
import matlab.engine

import matplotlib.pyplot as plt

import copy

class SpaceTimeGrid:

    def __init__(self, 
        paths: list[list[np.array]], r: float, dt: list[float],
        a_max: list[float], gamma: list[float], priority: list[float], 
        obs_paths: list[list[np.array]]=[], obs_dt: list[float]=[],
        tol: float=1e-5
    ) -> None:
        self.paths = [
            [np.append(s, dt[p_i] * i) for i, s in enumerate(p)] # add time dimension to all waypoints
        for p_i, p in enumerate(paths)] 
        self.alpha = 1 / (math.sqrt(3) - 1) # scale to remove discretization error
        self.r = self.alpha * r # radius of waypoints on paths
        self.dt = dt # dt of each agent planner
        self.a_max = a_max # max acceleration of agent paths
        self.gamma = gamma # flexibility constant of agent paths
        self.priority = priority # priority of agent paths
        self.obs_paths = [
            [np.append(s, obs_dt[p_i] * i) for i, s in enumerate(p)] # add time dimension to all waypoints
        for p_i, p in enumerate(obs_paths)] 
        if len(self.obs_paths) == 0:
            self.obs_paths.append([np.array([-1e9, -1e9, -1e9])])
        self.obs_dt = obs_dt # dt of each obs planner
        self.vel = [[np.array([0, 0]) for _ in p] for p in self.paths] # velocity at each waypoint for agent paths
        self.at_goal = [False for i in range(len(self.paths))] # whether agent has reached goal
        self.tree = KDTree([s for p in self.paths for s in p])
        self.obs_tree = KDTree([s for p in self.obs_paths for s in p])
        self.tol = tol
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r"C:\Users\aniru\OneDrive\Documents\Code\ICL\OptimizationSolver", nargout=0)

        # performance
        self.opt_num = 0
        self.opt_time = 0
        self.tree_time = 0
        self.res_num = 0

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

    def _common_vis(self, vis, vis1, vis2):
        for p_i in range(len(self.paths)):
            for s_i in range(len(self.paths[p_i])):
                if not vis[p_i][s_i] and vis1[p_i][s_i] and vis2[p_i][s_i]:
                    return True
        return False

    def _solve(self, T, p_i, s_i, v, vis, vis_obs, shifts, conflict):

        print(f"CURRENT: {p_i} {s_i} {v} {vis[p_i][s_i]}")
        print(f"VIS CURRENT: {vis}")
        
        log_list = []
        V_list = []
        vis_list = []
        vis_obs_list = []
        shifts_list = []
        if vis[p_i][s_i]:
            print(f" {p_i} {s_i} INFEASIBLE")
            return [], [], [], [], []
        vis_new = copy.deepcopy(vis)
        vis_new[p_i][s_i] = True
        T_new = np.asarray(copy.deepcopy(T))
        T_new[p_i], vel_new, sh = self._path_shift(T, p_i, s_i, v)
        shifts_new = np.asarray(copy.deepcopy(shifts))
        for i in range(len(shifts_new[p_i])):
            shifts_new[p_i][i] += sh[i]

        print(f"NEW ({p_i} {s_i}) LOCATION: {T_new[p_i][s_i]}")

        start = time.time()
        local_tree = KDTree([s for p in T_new for s in p])
        end = time.time()
        self.tree_time += end - start

        # if initial conflict is unresolved post-shift, then infeasible
        p_con, s_con, is_ob_con = conflict
        if is_ob_con:
            query_obs = self.obs_tree.query_ball_point(T_new[p_i][s_i], self.r + self.r - self.tol)
            inter_obs = [tuple(self._idx2sp(self.obs_paths, idx)) for idx in query_obs]
            if (p_con, s_con) in inter_obs:
                print(f" {p_i} {s_i} INFEASIBLE")
                return [], [], [], [], []
        else:
            query = local_tree.query_ball_point(T_new[p_i][s_i], self.r + self.r - self.tol)
            inter = [tuple(self._idx2sp(self.paths, idx)) for idx in query]
            if (p_con, s_con) in inter:
                print(f" {p_i} {s_i} INFEASIBLE")
                return [], [], [], [], []

        # agent-agent intersection
        feasible = False
        not_empty = False
        query = local_tree.query_pairs(self.r + self.r - self.tol)
        for idx1, idx2 in query:
            p_j, s_j = self._idx2sp(T_new, idx1)
            p_k, s_k = self._idx2sp(T_new, idx2)
            if p_j == p_k:
                continue
            not_empty = True
            print(f"{p_i} {s_i} AA QUERY: ({p_j} {s_j}), ({p_k} {s_k})")
            print(f"AA QUERY LOCATIONS: {T_new[p_j][s_j]} {T_new[p_k][s_k]}")
            v1 = self._compute_dv(T_new[p_j][s_j], T_new[p_k][s_k])
            v2 = self._compute_dv(T_new[p_k][s_k], T_new[p_j][s_j])
            print(f"{p_i} {s_i} CALLS {p_j} {s_j}")
            print(f"VIS SENT ON CALL: {vis_new}")
            log1, V1, vis1, vis_obs1, shifts1 = self._solve(T_new, p_j, s_j, v1, vis_new, vis_obs, shifts_new, (p_k, s_k, False))
            print(f"{p_i} {s_i} CALLS {p_k} {s_k}")
            print(f"VIS SENT ON CALL: {vis_new}")
            log2, V2, vis2, vis_obs2, shifts2 = self._solve(T_new, p_k, s_k, v2, vis_new, vis_obs, shifts_new, (p_j, s_j, False))
            if len(log1) + len(log2) > 0:
                feasible = True
            paired1 = [False for _ in log1]
            paired2 = [False for _ in log2]
            for i in range(len(log1)):
                for j in range(len(log2)):
                    print(f"COMMON VIS ({p_j} {s_j}) ({p_k} {s_k}): {self._common_vis(vis_new, vis1[i], vis2[j])}")
                    if self._common_vis(vis_new, vis1[i], vis2[j]):
                        continue
                    paired1[i] = True
                    paired2[j] = True
                    log_list.append(log1[i] + log2[j])
                    V_list.append(V1[i] + V2[j])
                    vis_combo = [[vis1[i][p_idx][s_idx] or vis2[j][p_idx][s_idx] 
                            for s_idx in range(len(self.paths[p_idx]))] 
                                for p_idx in range(len(self.paths))]
                    vis_obs_combo = [[vis_obs1[i][p_idx][s_idx] or vis_obs2[j][p_idx][s_idx]
                            for s_idx in range(len(self.obs_paths[p_idx]))] 
                                for p_idx in range(len(self.obs_paths))]
                    vis_list.append(vis_combo)
                    vis_obs_list.append(vis_obs_combo)
                    shifts_combo = [shifts_new[idx] + shifts1[i][idx] + shifts2[j][idx] 
                        for idx in range(len(shifts_new))]
                    shifts_list.append(shifts_combo)
            for i in range(len(log1)):
                if paired1[i]:
                    continue
                if not vis1[i][p_k][s_k]:
                    log1[i] += [(p_k, s_k, False)]
                    V1[i] += [np.zeros(3)]
                    vis1[i][p_k][s_k] = True
                log_list.append(log1[i])
                V_list.append(V1[i])
                vis_list.append(vis1[i])
                vis_obs_list.append(vis_obs1[i])
                shifts_list.append([shifts_new[idx] + shifts1[i][idx] 
                    for idx in range(len(shifts_new))])
            for i in range(len(log2)):
                if paired2[i]:
                    continue
                if not vis2[i][p_j][s_j]:
                    log2[i] += [(p_j, s_j, False)]
                    V2[i] += [np.zeros(3)]
                    vis2[i][p_j][s_j] = True
                log_list.append(log2[i])
                V_list.append(V2[i])
                vis_list.append(vis2[i])
                vis_obs_list.append(vis_obs2[i])
                shifts_list.append([shifts_new[idx] + shifts2[i][idx] 
                    for idx in range(len(shifts_new))])
        if not feasible and not_empty:
            print(f"{p_i} {s_i} INFEASIBLE")
            return [], [], [], [], [] # no solution

        # agent-obs intersection
        feasible_obs = False
        not_empty_obs = False
        query_obs = local_tree.query_ball_tree(self.obs_tree, self.r + self.r - self.tol)
        print(query_obs)
        for j in range(len(query_obs)):
            for k_i in range(len(query_obs[j])):
                k = query_obs[j][k_i]
                not_empty_obs = True
                p_j, s_j = self._idx2sp(T_new, j)
                p_k, s_k = self._idx2sp(self.obs_paths, k)
                print(f"{p_i} {s_i} AO QUERY: ({p_j} {s_j}), ({p_k} {s_k})")
                vis_obs_new = np.asarray(copy.deepcopy(vis_obs))
                vis_obs_new[p_k][s_k] = True
                doubled = False
                p_next, s_next = None, None
                if k_i < len(query_obs[j]) - 1 and query_obs[j][k_i + 1] == k + 1:
                    doubled = True
                    p_next, s_next = self._idx2sp(self.obs_paths, k + 1)
                    avg = 0.5 * (self.obs_paths[p_k][s_k] + self.obs_paths[p_next][s_next])
                    v1 = self._compute_dv(T_new[p_j][s_j], avg)
                    vis_obs_new[p_next][s_next] = True
                    k_i += 1
                else:
                    v1 = self._compute_dv(T_new[p_j][s_j], self.obs_paths[p_k][s_k])
                print(f"{p_i} {s_i} CALLS {p_j} {s_j}")
                log1, V1, vis1, vis_obs1, shifts1 = self._solve(T_new, p_j, s_j, v1, vis_new, vis_obs_new, shifts_new, (p_k, s_k, True))
                if len(log1) > 0:
                    feasible_obs = True
                for i in range(len(log1)):
                    log_list.append(log1[i] + [(p_k, s_k, True)])
                    V_list.append(V1[i] + [np.zeros(3)])
                    if doubled:
                        log_list[-1].append((p_next, s_next, True))
                        V_list[-1].append(np.zeros(3))
                    vis_list.append(vis1[i])
                    vis_obs_list.append(vis_obs1[i])
                    shifts_list.append([shifts_new[idx] + shifts1[i][idx] 
                        for idx in range(len(shifts_new))])
                if len(log1) == 0:
                    log_list.append([(p_k, s_k, True)])
                    V_list.append([np.zeros(3)])
                    if doubled:
                        log_list[-1].append((p_next, s_next, True))
                        V_list[-1].append(np.zeros(3))
                    vis_list.append(vis_new)
                    vis_obs_list.append(vis_obs_new)
                    shifts_list.append(shifts_new)
        if not feasible_obs and not_empty_obs:
            print(f"{p_i} {s_i} INFEASIBLE")
            return [], [], [], [], [] # no solution

        for i in range(len(log_list)):
            # if not vis_list[i][p_i][s_i]:
            log_list[i].append((p_i, s_i, False))
            V_list[i].append(v)

        if len(log_list) == 0:
            log_list.append([(p_i, s_i, False)])
            V_list.append([v])
            vis_list.append(vis_new)
            vis_obs_list.append(vis_obs)
            shifts_list.append(shifts_new)

        print(f"{p_i} {s_i} FEASIBLE: {log_list}")
        return log_list, V_list, vis_list, vis_obs_list, shifts_list

    def _simulate(self):
        
        S_list = []
        V_list = []
        log_list = []

        # initially intersecting agent-agent
        query = self.tree.query_pairs(self.r + self.r - self.tol)
        for idx1, idx2 in query:
            p_i, s_i = self._idx2sp(self.paths, idx1)
            p_j, s_j = self._idx2sp(self.paths, idx2)
            if p_i == p_j:
                continue
            print(f"INITIAL AGENT-AGENT INTERSECTION: {self.paths[p_i][s_i]} {self.paths[p_j][s_j]}")
            v1 = self._compute_dv(self.paths[p_i][s_i], self.paths[p_j][s_j])
            v2 = self._compute_dv(self.paths[p_j][s_j], self.paths[p_i][s_i])
            vis1_init = np.array([[False for _ in p] for p in self.paths])
            # vis1_init[p_j][s_j] = True
            vis2_init = np.array([[False for _ in p] for p in self.paths])
            # vis2_init[p_i][s_i] = True
            vis_obs = np.array([[False for _ in p] for p in self.obs_paths])
            shifts = np.array([[np.zeros(3) for _ in p] for p in self.paths])
            print("AGENT-AGENT", p_i, s_i, v1, p_j, s_j, v2)
            print(f"SOLVE {p_i} {s_i} {v1}")
            log1, V1, vis1, vis_obs1, shifts1 = self._solve(self.paths, p_i, s_i, v1, vis1_init, vis_obs, shifts, (p_j, s_j, False))
            print(f"SOLVE {p_j} {s_j} {v2}")
            log2, V2, vis2, vis_obs2, shifts2 = self._solve(self.paths, p_j, s_j, v2, vis2_init, vis_obs, shifts, (p_i, s_i, False))
            paired1 = [False for _ in log1]
            paired2 = [False for _ in log2]
            for i in range(len(log1)):
                for v_i in range(len(V1[i])):
                    p_k, s_k, is_ob = log1[i][v_i]
                    if not is_ob:
                        V1[i][v_i] += shifts1[i][p_k][s_k]
            for i in range(len(log2)):
                for v_i in range(len(V2[i])):
                    p_k, s_k, is_ob = log2[i][v_i]
                    if not is_ob:
                        V2[i][v_i] += shifts2[i][p_k][s_k]
            for i in range(len(log1)):
                for j in range(len(log2)):
                    if self._common_vis([[False for _ in p] for p in self.paths], vis1[i], vis2[j]):
                        continue
                    print("PAIRING")
                    paired1[i] = True
                    paired2[j] = True
                    log_list.append(log1[i] + log2[j])
                    V_list.append(V1[i] + V2[j])
                    S_list.append([])
                    for p_k, s_k, is_ob in log_list[-1]:
                        if is_ob:
                            S_list[-1].append(self.obs_paths[p_k][s_k])
                        else:
                            S_list[-1].append(self.paths[p_k][s_k])
                    print(f"log_pair_1: {log1[i]}")
                    print(f"log_pair_2: {log2[j]}")
            for i in range(len(log1)):
                if not paired1[i]:
                    log_list.append(log1[i])
                    V_list.append(V1[i])
                    if not vis1[i][p_j][s_j]:
                        log_list[-1].append((p_j, s_j, False))
                        V_list[-1].append(np.zeros(3))
                    S_list.append([])
                    for p_k, s_k, is_ob in log_list[-1]:
                        if is_ob:
                            S_list[-1].append(self.obs_paths[p_k][s_k])
                        else:
                            S_list[-1].append(self.paths[p_k][s_k])
            for i in range(len(log2)):
                if not paired2[i]:
                    log_list.append(log2[i])
                    V_list.append(V2[i])
                    if not vis2[i][p_i][s_i]:
                        log_list[-1].append((p_i, s_i, False))
                        V_list[-1].append(np.zeros(3))
                    S_list.append([])
                    for p_k, s_k, is_ob in log_list[-1]:
                        if is_ob:
                            S_list[-1].append(self.obs_paths[p_k][s_k])
                        else:
                            S_list[-1].append(self.paths[p_k][s_k])

        # initially intersecting agent-obs
        query_obs = self.tree.query_ball_tree(self.obs_tree, self.r + self.r - self.tol)
        print(query_obs)
        for i in range(len(query_obs)):
            for j_i in range(len(query_obs[i])):
                j = query_obs[i][j_i]
                p_i, s_i = self._idx2sp(self.paths, i)
                p_j, s_j = self._idx2sp(self.obs_paths, j)
                print(f"INITIAL AGENT-OBS INTERSECTION: {self.paths[p_i][s_i]} {self.obs_paths[p_j][s_j]}")
                doubled = False
                p_next, s_next = None, None
                if j_i < len(query_obs[i]) - 1 and query_obs[i][j_i + 1] == j + 1:
                    doubled = True
                    p_next, s_next = self._idx2sp(self.obs_paths, j + 1)
                    avg = 0.5 * (self.obs_paths[p_j][s_j] + self.obs_paths[p_next][s_next])
                    v1 = self._compute_dv(self.paths[p_i][s_i], avg)
                    j_i += 1
                else:
                    v1 = self._compute_dv(self.paths[p_i][s_i], self.obs_paths[p_j][s_j])
                v2 = np.zeros(3)
                vis = [[False for _ in p] for p in self.paths]
                vis_obs = [[False for _ in p] for p in self.obs_paths]
                shifts = [[np.zeros(3) for _ in p] for p in self.paths]
                print("AGENT-OBS", p_i, s_i, v1, p_j, s_j, v2)
                print(f"SOLVE {p_i} {s_i} {v1}")
                log1, V1, vis1, vis_obs1, shifts1 = self._solve(self.paths, p_i, s_i, v1, vis, vis_obs, shifts, (p_j, s_j, True))
                for k in range(len(log1)):
                    for v_k in range(len(V1[k])):
                        p_k, s_k, is_ob = log1[k][v_k]
                        if not is_ob:
                            V1[k][v_k] += shifts1[k][p_k][s_k]
                for k in range(len(log1)):
                    log_list.append(log1[k])
                    V_list.append(V1[k])
                    if not vis_obs1[k][p_j][s_j]:
                        log_list[-1].append((p_j, s_j, True))
                        V_list[-1].append(np.zeros(3))
                    if doubled and not vis_obs1[k][p_next][s_next]:
                        log_list[-1].append((p_next, s_next, True))
                        V_list[-1].append(np.zeros(3))
                    S_list.append([])
                    for p_k, s_k, is_ob in log_list[-1]:
                        p_k, s_k = int(p_k), int(s_k)
                        if is_ob:
                            S_list[-1].append(self.obs_paths[p_k][s_k])
                        else:
                            S_list[-1].append(self.paths[p_k][s_k])

        return S_list, V_list, log_list

    def _path_shift(self, T, p_i, s_i, dv):

        shifts = [np.zeros(3) for _ in T[p_i]]

        if np.all(dv == 0):
            return T[p_i], self.vel[p_i], shifts

        p = np.asarray(copy.deepcopy(T[p_i]))
        s0 = p[s_i]
        d_max = max([np.linalg.norm(s - s0) for s in p])

        # apply path shift
        for i, s in enumerate(p):
            if i == 0: # don't move first sphere
                continue
            d = np.linalg.norm(s0 - s)
            mu = math.exp(-self.gamma[p_i] * (d / d_max)**2)
            vec = mu * dv
            if self.at_goal[p_i] and i == len(p) - 1:
                vec = vec[2] * np.array([0, 0, 1]) # only t component
            p[i] += vec
            if i != s_i:
                shifts[i] += vec

        # resolve time inversions and motion constraints
        vel = [np.zeros(2) for _ in p]
        for i in range(1, len(p)):
            delt = self._deltat(p[i - 1], p[i], vel[i - 1], self.a_max[p_i]) # time required to go from i-1 to i given v and a
            vel[i] = vel[i - 1] + self.a_max[p_i] * delt
            t_orig = p[i][2]
            p[i][2] = max(p[i][2], p[i - 1][2] + delt)
            shifts[i][2] += p[i][2] - t_orig

        # p, vel = self._pseudo_connect(p, vel) # TODO: incorporate this without messing up indexing in simulation

        return p, vel, shifts
        
    def _deltat(self, s1, s2, v1, a):
        d = np.linalg.norm(s2 - s1)
        v = abs((v1.T @ (s2[:2] - s1[:2])) / d) # component of velocity in direction of path
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
        return np.array(X), np.float64(fval)

    def _pseudo_connect(self, p, vel, end=False, clean=True):

        p_ret = list(copy.deepcopy(p))
        vel_ret = list(copy.deepcopy(vel))

        if end:
            # insert pseudo-waypoints between p[-2] and p[-1]
            num = math.ceil(np.linalg.norm(p[-1] - p[-2]) / (self.r + self.r)) - 1
            v_last = vel[-2]
            v_end = vel[-1]
            for i in range(num):
                # insert tangent to second to last sphere in direction towards last sphere
                mag = self.r + self.r
                uv = (p_ret[-1] - p_ret[-2]) / np.linalg.norm(p_ret[-1] - p_ret[-2])
                pseu = p_ret[-2] + mag * uv
                p_ret.insert(-1, pseu)
                vel_ret.insert(-1, v_last + v_end / num)
                v_last = vel_ret[-1]

        else:
            # insert pseudo-waypoints anywhere necessary
            p_new = []
            vel_new = []
            for i in range(len(p_ret)):
                p_new.append(p_ret[i])
                vel_new.append(vel_ret[i])
                if i == len(p_ret) - 1:
                    continue
                num = math.ceil(np.linalg.norm(p_ret[i + 1] - p_ret[i]) / (self.r + self.r)) - 1
                s1 = p_ret[i]
                s2 = p_ret[i + 1]
                v1 = vel_ret[i]
                v2 = vel_ret[i + 1]
                for j in range(num):
                    mag = self.r + self.r
                    uv = (s2 - s1) / np.linalg.norm(s2 - s1)
                    pseu = s1 + mag * uv
                    p_new.append(pseu)
                    vel_new.append(v1 + v2 / num)
                    s1 = pseu
                    v1 = vel_new[-1]
            p_ret = list(copy.deepcopy(p_new))
            vel_ret = list(copy.deepcopy(vel_new))

        # Remove unnecessary pseudo-waypoints:
        # iterate thru path, for each p[i] find the 
        # first p[j] that doesn't intersect p[i] where 
        # j > i, then delete everything between i and j - 1
        if clean:
            p_new = []
            vel_new = []
            i = 0
            while i < len(p_ret):
                p_new.append(p_ret[i])
                vel_new.append(vel_ret[i])
                if i == len(p_ret) - 1:
                    break
                j = i + 1
                while j < len(p_ret) and self._intersects(p_ret[i], p_ret[j], eps=1e-5):
                    j += 1
                i = j - 1
            p_ret = list(copy.deepcopy(p_new))
            vel_ret = list(copy.deepcopy(vel_new))

        return p_ret, vel_ret

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

        self.paths[p_i], self.vel[p_i] = self._pseudo_connect(self.paths[p_i], self.vel[p_i], end=True, clean=False)

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

    def resolve(self, i) -> None: # TODO: delete i param after debug

        if i >= 10:
            # plot paths
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            c = ["blue", "red", "green", "orange"]
            for i, p in enumerate(self.paths + self.obs_paths):
                ax.scatter([s[0] for s in p], [s[1] for s in p], [s[2] for s in p], color=c[i])
            plt.show()

        min_clear = np.inf
        for i, p1 in enumerate(self.paths + self.obs_paths):
            for p2 in (self.paths + self.obs_paths)[i + 1:]:
                for s1 in p1:
                    for s2 in p2:
                        min_clear = min(min_clear, np.linalg.norm(s1 - s2) / self.alpha)
        print(f"MIN CLEARANCE BEFORE: {min_clear}")

        S_list, V_list, log_list = self._simulate()

        print('\n')
        for S, V, log in zip(S_list, V_list, log_list):
            for i in range(len(S)):
                print(f"S[i]: {S[i]} ##### V[i]: {V[i]} ##### log[i]: {log[i]}")
            print("---------------------------------------------------")

        X_star = []
        min_f = np.inf
        best_i = -1
        for i, (S, V, log) in enumerate(zip(S_list, V_list, log_list)):
            S, V = np.array(S), np.array(V)
            P = self._get_P(S, log)
            pri = np.array([1 if is_ob else self.priority[p_i]for p_i, _, is_ob in log])
            rad = np.array([self.r for i in range(S.shape[0])])
            X, fval = self._optimize(S, V, P, pri, rad)
            if fval < min_f:
                X_star = X
                min_f = fval
                best_i = i

        # plot paths and outstanding spheres
        if len(S_list) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            c = ["blue", "red", "green", "orange"]
            for i, p in enumerate(self.paths + self.obs_paths):
                ax.scatter([s[0] for s in p], [s[1] for s in p], [s[2] for s in p], color=c[i])
            ax.scatter(
                [s[0] for s in S_list[best_i]], [s[1] for s in S_list[best_i]], [s[2] for s in S_list[best_i]], 
            color="yellow", s=100)
            plt.show()

        if len(S_list) > 0:
            for (p_i, s_i, is_ob), x in zip(log_list[best_i], X_star):
                if is_ob:
                    continue
                vec = x - self.paths[p_i][s_i]
                self.paths[p_i], self.vel[p_i], shifts = self._path_shift(self.paths, p_i, s_i, vec)
            self.res_num += 1

        for p_i in range(len(self.paths)):
            self.paths[p_i], self.vel[p_i] = self._pseudo_connect(self.paths[p_i], self.vel[p_i])

        start = time.time()
        self.tree = KDTree([s for p in self.paths for s in p]) # reinitialize tree
        end = time.time()
        self.tree_time += end - start

        min_clear = np.inf
        for i, p1 in enumerate(self.paths + self.obs_paths):
            for p2 in (self.paths + self.obs_paths)[i + 1:]:
                for s1 in p1:
                    for s2 in p2:
                        min_clear = min(min_clear, np.linalg.norm(s1 - s2) / self.alpha)
        print(f"MIN CLEARANCE AFTER: {min_clear}")

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
