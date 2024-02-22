class CBF_hF22:
    def __init__(self, x_init, s_min, L, dt, c_min, idm):
        self.s_min = s_min
        self.L = L
        self.dt = dt
        self.c_min = c_min
        self.idm = idm
        self.alpha = self._alpha(x_init)

    def h(self, x):
        return x['pR'][0] - x['pH'][0] - self.s_min - self.L
    def h_dot(self, x):
        return x['vR'][0] - x['vH'][0]
    def h_ddot(self, x):
        return x['aR'][0] - x['aH'][0]
    
    def lb(self, x):
        # u - f + ah_dot + ah >= 0
        # u >= f - ah_dot - ah
        return x['aH'][0] - self.alpha[1] * self.h_dot(x) - self.alpha[0] * self.h(x)
    
    def _alpha(self, x):
        c1 = max(0, -self.h_dot(x) / self.h(x)) + self.c_min
        c2 = max(0, -(self.h_ddot(x) + c1 * self.h_dot(x)) 
                        / (self.h_dot(x) + c1 * self.h(x))) + self.c_min
        return [c1 + c2, c1 * c2]