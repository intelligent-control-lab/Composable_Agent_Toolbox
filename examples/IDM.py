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
        return -self.a * (s_star / (xR - xH - self.L))**2

    def idm(self, xH, xR, vH, vR):
        return self.free_road(vH) + self._interaction_term(xH, xR, vH, vR)

    def dIdm_dvR(self, xH, xR, vH, vR, aH, aR):
        dxH_dvR = vH / aR
        dxR_dvR = vR / aR
        dvH_dvR = aH / aR
        return self.a * (-(2 * ((vH * (vH - vR)) / (2 * math.sqrt(self.a * self.b)) + 
            self.s0 + self.T * vH) * ((vH * (dvH_dvR - 1)) / (2 * math.sqrt(self.a * self.b)) + 
            ((vH - vR) * dvH_dvR) / (2 * math.sqrt(self.a * self.b)) + self.T * dvH_dvR)) / 
            (-self.L - xH + xR)**2 + (2 * (dxR_dvR - dxH_dvR) * ((vH * (vH - vR)) / 
            (2 * math.sqrt(self.a * self.b)) + self.s0 + self.T * vH)**2) / 
            (-self.L - xH + xH)**3 - (4 * vH**3 * dvH_dvR) / self.v0**4)
