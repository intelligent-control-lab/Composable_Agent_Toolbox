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

    def _interaction_term(self, xH, xR, vH, vR):
        s_star = self.s0 + vH * self.T + (vH * (vH - vR)) / \
            (2 * math.sqrt(self.a * self.b))
        return -self.a * (max(0, s_star) / (xR - xH - self.L))**2

    def idm(self, xH, xR, vH, vR):
        return self.free_road(vH) + self._interaction_term(xH, xR, vH, vR)
    
    def df_dxH(self, xH, xR, vH, vR):
        return (2 * math.sqrt(self.a * self.b) * (self.s0 + self.T * vH) 
                + vH * (vH - vR))**2 / (2 * self.b * (self.L + xH - xR)**3)
    def df_dxR(self, xH, xR, vH, vR):
        return -(2 * math.sqrt(self.a * self.b) * (self.s0 + self.T * vH) 
                + vH * (vH - vR))**2 / (2 * self.b * (self.L + xH - xR)**3)
    def df_dvH(self, xH, xR, vH, vR):
        return -((2 * self.T * math.sqrt(self.a * self.b) + 2 * vH - vR) 
                 * (2 * self.s0 * math.sqrt(self.a * self.b) + vH * 
                    (2 * self.T * math.sqrt(self.a * self.b) + vH - vR))) \
                / (2 * self.b * (self.L + xH - xR)**2) - (4 * self.a * vH**3) / self.v0**4
    def df_dvR(self, xH, xR, vH, vR):
        return (vH * (2 * math.sqrt(self.a * self.b) 
                      * (self.s0 + self.T * vH) + vH * (vH - vR))) \
                    / (2 * self.b * (self.L + xH - xR)**2)

    def control_lower_bound(self, bH_max, bR_max, res, xH, xR, vH, vR, dt):
        lb = 0
        while lb > -bR_max:
            vR_new = vR + (lb - res) * dt
            xR_new = xR + vR_new * dt
            # print(lb - res, self.idm(xH, xR_new, vH, vR_new))
            if self.idm(xH, xR_new, vH, vR_new) < -bH_max:
                break
            lb -= res
        return lb
