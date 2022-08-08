import math
from queue import Queue

import numpy as np
from scipy.spatial import KDTree

import cvxpy as cp


spheres = []
tree = None # will need dynamic structure (insert/delete), perhaps ikd-tree (https://github.com/hku-mars/ikd-Tree)
r = 1 # try making heterogeneous in future

class Sphere:
    def __init__(self, pos, rad):
        self.pos = pos
        self.rad = rad

def compute_dv(s1, s2):
    mag = s1.rad + s2.rad - math.sqrt(
        (s1.pos[0] - s2.pos[0])**2 + (s1.pos[1] - s2.pos[1])**2 + (s1.pos[2] - s2.pos[2])**2)
    uv = (s1.pos - s2.pos) / np.linalg.norm(s1.pos - s2.pos)
    return mag * uv

def intersects(s1, s2):
    return math.sqrt(
        (s1.pos[0] - s2.pos[0])**2 + (s1.pos[1] - s2.pos[1])**2 + (s1.pos[2] - s2.pos[2])**2
    ) < s1.rad + s2.rad

# TODO: figure out: should sweep stop at first intersection or continue thru?
# is it possible to iterate by distance 2r instead of r? (prove all intersections are detected)
def sweep(s_cur, v_cur):

    # # OPTION 1: return "earliest" set
    # s_trans = Sphere(s_cur.pos, s_cur.rad)
    # # translate s_trans along v_cur
    # while np.linalg.norm(s_trans.pos - s_cur.pos) < np.linalg.norm(v_cur):
    #     inter = tree.query_ball_point(s_trans.pos, r + r)
    #     if len(inter) > 0:
    #         return set(spheres[i] for i in inter)
    #     s_trans.pos += r * v_cur / np.linalg.norm(v_cur) # iterate by distance r
    # return set()

    # OPTION 2: return entire set
    inter = set()
    s_trans = Sphere(s_cur.pos, s_cur.rad)

    # translate s_trans along v_cur
    while np.linalg.norm(s_trans.pos - s_cur.pos) < np.linalg.norm(v_cur):
        query = tree.query_ball_point(s_trans.pos, r + r)
        inter.update(set(spheres[i] for i in query))
        s_trans.pos = s_trans.pos + r * v_cur / np.linalg.norm(v_cur) # iterate by distance r

    s_trans.pos = s_cur.pos + v_cur # iterate to end of vector
    query = tree.query_ball_point(s_trans.pos, r + r)
    inter.update(set(spheres[i] for i in query))
    inter.remove(s_cur)
    return inter

# even if s_cur sweeps all the way past some sphere, 
# that sphere still has to move "past" s_cur for its DV

def simulate(s1, s2):
    outstanding = []
    q = Queue()
    v1 = compute_dv(s1, s2)
    v2 = compute_dv(s2, s1)
    q.put((s1, v1))
    q.put((s2, v2))
    while not q.empty():
        s_cur, v_cur = q.get()
        outstanding.append((s_cur, v_cur))
        for s_new in sweep(s_cur, v_cur):
            if (s_cur == s1 and s_new == s2) or (s_cur == s2 and s_new == s1): # prevent infinite loop
                continue
            s_trans = Sphere(s_cur.pos + v_cur, s_cur.rad)
            v_new = compute_dv(s_new, s_trans)
            q.put((s_new, v_new))
    return outstanding


if __name__ == '__main__':

    # https://www.geogebra.org/3d/ma7vpx3m
    points = np.array([
        np.array([2.5, -0.5, 2.5]),
        np.array([0.0, 2.0, 1.0]),
        np.array([-1.0, 0.0, 4.0]),
        np.array([1.0, -2.0, 2.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([0.5, -3.0, 1.5]),
        np.array([-1.0, -3.0, 0.0])
    ])

    tree = KDTree(points)
    for p in points:
        spheres.append(Sphere(p, r))

    inter = tree.query_pairs(r + r).pop()
    outstanding = simulate(spheres[inter[0]], spheres[inter[1]])
    for s, v in outstanding:
        print(f'({spheres.index(s) + 1}) {s.pos} : {v}')

    n = len(points)
    x = cp.Variable((n, 3)) # coordinates of each sphere

    # objective function: minimize total displacement of all points
    # perhaps minimize displacement ALONG respective DVs instead?
    obj = cp.Minimize(cp.sum(cp.norm(x - points, 2, axis=1)))

    constraints = [
        # constraint 1
        cp.norm(x - points) <=  # constraint 2
        # constraint 3
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve()
    print(x.value)
