import math
import numpy as np
import uncertainties

def _intersect(pH, vH, pR, vR):
    if vH[0]*vR[1]-vR[0]*vH[1] == 0:
        return None
    t = (vR[1]*(pR[0]-pH[0])-vR[0]*(pR[1]-pH[1])) / (vH[0]*vR[1]-vR[0]*vH[1])
    return np.array([pH[0]+vH[0]*t, pH[1]+vH[1]*t])

def collision_dist2(x, m, infra, agent): # worst-case min dist2 from collision w any obs
    min_dist2 = np.inf
    for i in range(m):
        bel_p = infra.bel[agent][i]['pos']
        bel_v = infra.bel[agent][i]['vel']
        pH = np.array([uncertainties.ufloat(bel_p[0][0], math.sqrt(bel_p[1][0])), 
                       uncertainties.ufloat(bel_p[0][1], math.sqrt(bel_p[1][1]))])
        vH = np.array([uncertainties.ufloat(bel_v[0][0], math.sqrt(bel_v[1][0])), 
                       uncertainties.ufloat(bel_v[0][1], math.sqrt(bel_v[1][1]))])
        inter = _intersect(pH, vH, x['pR'][agent], x['vR'][agent])
        if inter is not None:
            dist2_u = (pH - inter).T @ (pH - inter)
            print("DIST2", dist2_u)
            dist2 = dist2_u.n - dist2_u.s # 1 stdev below mean
            min_dist2 = min(min_dist2, dist2)
    return max(0, min_dist2)