class CBF_hF31:
    def __init__(self, x_init, s_min, L, dt, c_min, idm):
        self.s_min = s_min
        self.L = L
        self.dt = dt
        self.c_min = c_min
        self.idm = idm
        self.alpha = self._alpha(x_init)

    def h(self, x):
        return x['pR'][0] - x['pH'][1] - self.s_min - self.L
    def h_dot(self, x):
        return x['vR'][0] - x['vH'][1]
    def h_ddot(self, x):
        return x['aR'][0] - x['aH'][1]
    
    def constraint(self, x):
        print(f"F: {self.h(x)} {self.h_dot(x)} {self.h_ddot(x)}")
        coeff1 = self.alpha[2] + 1 / self.dt
        coeff2 = -self.idm.df_dvR(x['pH'][1], x['pR'][1], x['vH'][1], x['vR'][1])
        beta = x['aR'][0] / self.dt + self.idm.lamb(x['pH'][1], x['pR'][1], x['vH'][1], x['vR'][1], x['aH'][1]) + \
            self.alpha[2] * x['aH'][1] - self.alpha[1] * self.h_dot(x) - self.alpha[0] * self.h(x)
        # coeff1*u1 + coeff2*u2 >= beta
        return (coeff1, coeff2, beta) 
    
    def _alpha(self, x):
        c1 = max(0, -self.h_dot(x) / self.h(x)) + self.c_min
        c2 = max(0, -(self.h_ddot(x) + c1 * self.h_dot(x)) 
                        / (self.h_dot(x) + c1 * self.h(x))) + self.c_min
        c3 = max(0, -(0 + c2 * self.h_ddot(x) + c1 * self.h_dot(x)) 
                / (self.h_ddot(x) + c2 * self.h_dot(x) + c1 * self.h(x))) + self.c_min
        return [c1 + c2 + c3, 
                c1*c2 + c1*c3 + c2*c3,
                c1 * c2 * c3]