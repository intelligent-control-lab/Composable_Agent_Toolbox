class CBF_hD2_33:
    def __init__(self, x_init, s_min, L, dt, c_min, idm):
        self.s_min = s_min
        self.L = L
        self.dt = dt
        self.c_min = c_min
        self.idm = idm
        self.alpha = self._alpha(x_init)

    def h(self, x):
        return x['pH'][0] - x['pB'][0] - self.s_min - self.L
    def h_dot(self, x):
        return x['vH'][0] - x['vB'][0]
    def h_ddot(self, x):
        return x['aH'][0]
    
    def lb(self, x): # u2 >= lb
        # print(f"D2: {self.idm.lamb(x['pH'][1], x['pR'][1], x['vH'][1], x['vR'][1], x['aH'][1])} {self.alpha[2]} {self.h_ddot(x)} {self.alpha[1]} {self.h_dot(x)} {self.alpha[0]} {self.h(x)} {self.idm.df_dvR(x['pH'][1], x['pR'][1], x['vH'][1], x['vR'][1])}")
        return (-self.idm.lamb(x['pH'][0], x['pR'][0], x['vH'][0], x['vR'][0], x['aH'][0]) 
          - self.alpha[2] * self.h_ddot(x) - self.alpha[1] * self.h_dot(x) - self.alpha[0] * self.h(x)) \
        / -self.idm.df_dvR(x['pH'][0], x['pR'][0], x['vH'][0], x['vR'][0])
    
    def _alpha(self, x):
        c1 = max(0, -self.h_dot(x) / self.h(x)) + self.c_min
        c2 = max(0, -(self.h_ddot(x) + c1 * self.h_dot(x)) 
                        / (self.h_dot(x) + c1 * self.h(x))) + self.c_min
        c3 = max(0, -(0 + c2 * self.h_ddot(x) + c1 * self.h_dot(x)) 
                / (self.h_ddot(x) + c2 * self.h_dot(x) + c1 * self.h(x))) + self.c_min
        return [c1 + c2 + c3, 
                c1*c2 + c1*c3 + c2*c3,
                c1 * c2 * c3]