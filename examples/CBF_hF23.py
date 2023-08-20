class CBF_hF23:
    def __init__(self, x_init, s_min, L, dt, c_min, idm):
        self.s_min = s_min
        self.L = L
        self.dt = dt
        self.c_min = c_min
        self.idm = idm
        self.alpha = self._alpha(x_init)

    def h(self, x):
        return x['pB'][0] - x['pH'][0] - self.s_min - self.L
    def h_dot(self, x):
        return x['vB'][0] - x['vH'][0]
    def h_ddot(self, x):
        return -x['aH'][0]
    
    def ub(self, x): # u1 <= ub
        return -self.idm.lamb(x['pH'][0], x['pR'][0], x['vH'][0], x['vR'][0], x['aH'][0]) + \
            self.alpha[2] * self.h_ddot(x) + self.alpha[1] * self.h_dot(x) + self.alpha[0] * self.h(x)
    
    def _alpha(self, x):
        c1 = max(0, -self.h_dot(x) / self.h(x)) + self.c_min * 1000
        c2 = max(0, -(self.h_ddot(x) + c1 * self.h_dot(x)) 
                        / (self.h_dot(x) + c1 * self.h(x))) + self.c_min * 1000
        c3 = max(0, -(0 + c2 * self.h_ddot(x) + c1 * self.h_dot(x)) 
                / (self.h_ddot(x) + c2 * self.h_dot(x) + c1 * self.h(x))) + self.c_min * 1000
        return [c1 + c2 + c3, 
                c1*c2 + c1*c3 + c2*c3,
                c1 * c2 * c3]