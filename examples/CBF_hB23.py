class CBF_hB23:
    def __init__(self, x_init, s_min, L, dt, c_min, idm):
        self.s_min = s_min
        self.L = L
        self.dt = dt
        self.c_min = c_min
        self.idm = idm
        self.alpha = self._alpha(x_init)

    def h(self, x):
        return x['pH'][0] - x['pR'][1] - self.s_min - self.L
    def h_dot(self, x):
        return x['vH'][0] - x['vR'][1]
    def h_ddot(self, x):
        return x['aH'][0] - x['aR'][1]
    
    def constraint(self, x):
        coeff1 = self.idm.df_dvR(x['pH'][0], x['pR'][0], x['vH'][0], x['vR'][0])
        coeff2 = -(self.alpha[2] + 1 / self.dt)
        beta = -x['aR'][1] / self.dt - self.idm.lamb(x['pH'][0], x['pR'][0], x['vH'][0], x['vR'][0], x['aH'][0]) - \
            self.alpha[2] * x['aH'][0] - self.alpha[1] * self.h_dot(x) - self.alpha[0] * self.h(x)
        return (coeff1, coeff2, beta) 
    
    def _alpha(self, x):
        c1 = max(0, -self.h_dot(x) / self.h(x)) + self.c_min * 0.001
        c2 = max(0, -(self.h_ddot(x) + c1 * self.h_dot(x)) 
                        / (self.h_dot(x) + c1 * self.h(x))) + self.c_min * 0.001
        c3 = max(0, -(0 + c2 * self.h_ddot(x) + c1 * self.h_dot(x)) 
                / (self.h_ddot(x) + c2 * self.h_dot(x) + c1 * self.h(x))) + self.c_min * 0.001
        return [c1 + c2 + c3, 
                c1*c2 + c1*c3 + c2*c3,
                c1 * c2 * c3]