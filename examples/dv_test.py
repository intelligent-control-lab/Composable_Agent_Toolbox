import math
from queue import Queue
import time

import numpy as np
from scipy.spatial import KDTree

import matlab
import matlab.engine

spheres = [] # sphere centers
tree = None # will need dynamic structure (insert/delete), perhaps ikd-tree (https://github.com/hku-mars/ikd-Tree)
r = 1 # try making heterogeneous in future

def compute_dv(s1, s2):
    mag = r + r - math.sqrt(
        (s1[0] - s2[0])**2 + (s1[1] - s2[1])**2 + (s1[2] - s2[2])**2)
    uv = (s1 - s2) / np.linalg.norm(s1 - s2)
    return mag * uv

def intersects(s1, s2, epsilon=0):
    return math.sqrt(
        (s1[0] - s2[0])**2 + (s1[1] - s2[1])**2 + (s1[2] - s2[2])**2
    ) < r + r - epsilon

# TODO: figure out: should sweep stop at first intersection or continue thru?
# is it possible to iterate by distance 2r instead of r? (prove all intersections are detected)
def sweep(s_cur, v_cur):

    # OPTION 2: return entire set
    inter = []
    s_trans = s_cur.copy()

    # translate s_trans along v_cur
    while np.linalg.norm(s_trans - s_cur) < np.linalg.norm(v_cur):
        query = tree.query_ball_point(s_trans, r + r)
        for i in query:
            inter.append(spheres[i])
        s_trans = s_trans + r * v_cur / np.linalg.norm(v_cur) # iterate by distance r

    s_trans = s_cur + v_cur # iterate to end of vector
    query = tree.query_ball_point(s_trans, r + r)
    for i in query:
        inter.append(spheres[i])
    return inter

# even if s_cur sweeps all the way past some sphere, 
# that sphere still has to move "past" s_cur for its DV

def simulate():

    # difference between using bfs vs. dfs?

    # Currently for cases with multiple interdependent initial intersections,
    # each sphere just keeps the first DV that is computed for it, and future
    # ones are ignored. -> We need a better method of dealing with this, one
    # which we can prove guarantees a feasible solution through some linear
    # combination.

        # one way is to allow multiple DVs for each sphere until you reach a
        # "clone" DV, but this could become computationally expensive

        # this problem can be avoided if we just check intersections at every
        # new insertion

        # perhaps combine the vectors somehow?

    S = [] # outstanding spheres
    V = [] # DVs of S_out
    q = Queue()

    vis = {}
    for s in spheres:
        vis[tuple(s)] = False

    # initially intersecting
    for i, j in tree.query_pairs(r + r):
        print(i, j)
        s1, s2 = spheres[i], spheres[j]
        v1 = compute_dv(s1, s2)
        v2 = compute_dv(s2, s1)
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
        for s_new in sweep(s_cur, v_cur):
            if (s_new==s_cur).all() or ((s_cur==s1).all() and (s_new==s2).all()) or ((s_cur==s2).all() and (s_new==s1).all()): # prevent infinite loop
                continue
            s_trans = s_cur + v_cur
            v_new = compute_dv(s_new, s_trans)
            q.put((s_new, v_new))

    return S, V


def optimize(S, V, rad):

    print(f'Starting engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath(r"C:\Users\aniru\OneDrive\Documents\Code\ICL\OptimizationSolver", nargout=0)

    n = S.shape[0]
    S = matlab.double(S.tolist())
    V = matlab.double(V.tolist())
    rad = matlab.double(rad.tolist())

    print(f'Optimizing...')
    X, w, fval = eng.optimize(S, V, rad, n, nargout=3)

    print(f'X:\n{X}\n')
    print(f'w:\n{w}\n')
    print(f'fval:\n{fval}\n')

    input("Click Enter to finish...")
    eng.quit()

    return X

def check_feasible(X):
    print("Checking feasability...")
    feasible = True
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            s1, s2 = X[i], X[j]
            if intersects(s1, s2, epsilon=1e-7):
                feasible = False
                print(f"{i} intersects {j} in X")
    return feasible


if __name__ == '__main__':

    spheres = [
        np.array([2.5, -0.5, 2.5]),
        np.array([2.0, -0.5, 2.0]), # infeasible whenever a chain of 2+ spheres are initially intersecting
        np.array([0.0, 2.0, 1.0]),
        np.array([-1.0, 0.0, 4.0]),
        np.array([1.0, -2.0, 2.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([0.5, -3.0, 1.5]),
        np.array([-1.0, -3.0, 0.0])
    ]

    # spheres = [
    #     np.array([2.5, -0.5, 2.5]),
    #     np.array([3.0, 1.5, 3.0]),
    #     np.array([0.0, 2.0, 1.0]),
    #     np.array([-1.0, 0.0, 4.0]),
    #     np.array([1.0, -2.0, 2.0]),
    #     np.array([1.0, -1.0, 0.0]),
    #     np.array([0.5, -3.0, 1.5]),
    #     np.array([-1.0, -3.0, 0.0]),
    #     np.array([12.5, 9.5, 12.5]),
    #     np.array([10.0, 12.0, 11.0]),
    #     np.array([9.0, 10.0, 14.0]),
    #     np.array([11.0, 8.0, 12.0]),
    #     np.array([11.0, 9.0, 10.0]),
    #     np.array([10.5, 7.0, 11.5]),
    #     np.array([9.0, 7.0, 10.0])
    # ]

    # spheres = [
    #     np.array([-0.5, 0., 0.]),
    #     np.array([0.5, 0., 0.]),
    #     np.array([-3, 0., 0.]),
    #     np.array([3, 0., 0.])
    # ]
 
    tree = KDTree(spheres)

    print("Computing DVs...")
    S, V = simulate()
    for s_i, v_i in zip(S, V):
        print(f'{s_i} : {v_i}')

    S = np.array(S)
    V = np.array(V)
    rad = np.ones(S.shape[0])

    X = optimize(S, V, rad)

    print(f"Feasible: {check_feasible(X)}")
