import math
from queue import Queue

import numpy as np
from scipy.spatial import KDTree

import cvxpy as cp


spheres = [] # sphere centers
tree = None # will need dynamic structure (insert/delete), perhaps ikd-tree (https://github.com/hku-mars/ikd-Tree)
r = 1 # try making heterogeneous in future

def compute_dv(s1, s2):
    mag = r + r - math.sqrt(
        (s1[0] - s2[0])**2 + (s1[1] - s2[1])**2 + (s1[2] - s2[2])**2)
    uv = (s1 - s2) / np.linalg.norm(s1 - s2)
    return mag * uv

def intersects(s1, s2):
    return math.sqrt(
        (s1[0] - s2[0])**2 + (s1[1] - s2[1])**2 + (s1[2] - s2[2])**2
    ) < r + r

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

def simulate(s1, s2):

    S = [] # outstanding spheres
    V = [] # DVs of S_out
    q = Queue()
    v1 = compute_dv(s1, s2)
    v2 = compute_dv(s2, s1)
    q.put((s1, v1))
    q.put((s2, v2))

    while not q.empty():
        s_cur, v_cur = q.get()
        S.append(s_cur)
        V.append(v_cur)
        for s_new in sweep(s_cur, v_cur):
            if (s_new==s_cur).all() or ((s_cur==s1).all() and (s_new==s2).all()) or ((s_cur==s2).all() and (s_new==s1).all()): # prevent infinite loop
                continue
            s_trans = s_cur + v_cur
            v_new = compute_dv(s_new, s_trans)
            q.put((s_new, v_new))

    return np.array(S), np.array(V)


if __name__ == '__main__':

    # https://www.geogebra.org/3d/ma7vpx3m
    spheres = [
        np.array([2.5, -0.5, 2.5]),
        np.array([0.0, 2.0, 1.0]),
        np.array([-1.0, 0.0, 4.0]),
        np.array([1.0, -2.0, 2.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([0.5, -3.0, 1.5]),
        np.array([-1.0, -3.0, 0.0])
    ]

    tree = KDTree(spheres)
    inter = tree.query_pairs(r + r).pop() # get pair of intersecting spheres
    s1, s2 = spheres[inter[0]], spheres[inter[1]]

    S, V = simulate(s1, s2)
    for s_i, v_i in zip(S, V):
        print(f'{s_i} : {v_i}')

    x = cp.Variable(S.shape) # new coordinates of each sphere

    # objective function: minimize total displacement of all points
    # perhaps minimize displacement ALONG respective DVs instead?
    obj = cp.Minimize(cp.sum(cp.norm(x - S, axis=1)))

    constraints = [
        # # constraint 1
        # cp.norm(x - points) <=  # constraint 2
        # # constraint 3
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve()
    print(x.value)
