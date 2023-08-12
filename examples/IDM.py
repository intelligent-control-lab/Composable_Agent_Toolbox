import math


class IDM:
    def __init__(self, s0, v0, T, a, b, L):
        self.s0 = s0
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.L = L

    def free_road(self, vH):
        return self.a * (1 - (vH / self.v0)**4)

    def _interaction_term(self, pH, pR, vH, vR):
        s_star = self.s0 + vH * self.T + (vH * (vH - vR)) / \
            (2 * math.sqrt(self.a * self.b))
        return -self.a * (max(0, s_star) / (pR - pH - self.L))**2

    def idm(self, pH, pR, vH, vR):
        return self.free_road(vH) + self._interaction_term(pH, pR, vH, vR)
    
    def df_dpH(self, pH, pR, vH, vR):
        return (2 * math.sqrt(self.a * self.b) * (self.s0 + self.T * vH) 
                + vH * (vH - vR))**2 / (2 * self.b * (self.L + pH - pR)**3)
    def df_dpR(self, pH, pR, vH, vR):
        return -(2 * math.sqrt(self.a * self.b) * (self.s0 + self.T * vH) 
                + vH * (vH - vR))**2 / (2 * self.b * (self.L + pH - pR)**3)
    def df_dvH(self, pH, pR, vH, vR):
        return -((2 * self.T * math.sqrt(self.a * self.b) + 2 * (vH - vR)) 
                 * (2 * self.s0 * math.sqrt(self.a * self.b) + vH * 
                    (2 * self.T * math.sqrt(self.a * self.b) + vH - vR))) \
                / (2 * self.b * (self.L + pH - pR)**2) - (4 * self.a * vH**3) / self.v0**4
    def df_dvR(self, pH, pR, vH, vR):
        return (vH * (2 * math.sqrt(self.a * self.b) 
                      * (self.s0 + self.T * vH) + vH * (vH - vR))) \
                    / (2 * self.b * (self.L + pH - pR)**2)
    def lamb(self, pH, pR, vH, vR, aH):
        return self.df_dpH(pH, pR, vH, vR) * vH + self.df_dvH(pH, pR, vH, vR) * aH + self.df_dpR(pH, pR, vH, vR) * vR

    def control_lower_bound(self, bH_max, bR_max, res, pH, pR, vH, vR, dt):
        lb = 0
        while lb > -bR_max:
            vR_new = vR + (lb - res) * dt
            pR_new = pR + vR_new * dt
            # print(lb - res, self.idm(pH, pR_new, vH, vR_new))
            if self.idm(pH, pR_new, vH, vR_new) < -bH_max:
                break
            lb -= res
        return lb
