# inverse of min/avg dist between paths across horizon
import numpy as np

def heuristic(pH, vH, pR, vR):
    horizon = 5
    dt = 0.1
    min_dist2 = np.inf
    tot_dist2 = 0
    posH = pH.copy()
    posR = pR.copy()
    for t in np.arange(0, horizon, dt):
        posH += vH * dt
        posR += vR * dt
        dist2 = (posH - posR).T @ (posH - posR)
        min_dist2 = min(min_dist2, dist2)
        tot_dist2 += dist2
    return 1 / tot_dist2