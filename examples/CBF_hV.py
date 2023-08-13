class CBF_hV:
    def __init__(self, x_init, v_max, c_min, idm):
        self.v_max = v_max
        self.c_min = c_min
        self.idm = idm
        self.alpha = self._alpha(x_init)

    def h(self, x):
        return self.v_max - x['vH'][0]
    def h_dot(self, x):
        return -x['aH'][0]
    
    def ub(self, x):
        return (self.alpha[1] * self.h_dot(x) + self.alpha[0] * self.h(x) 
          - self.idm.lamb(x['pH'][0], x['pR'][0], x['vH'][0], x['vR'][0], x['aH'][0])) \
    / self.idm.df_dvR(x['pH'][0], x['pR'][0], x['vH'][0], x['vR'][0])
    
    def _alpha(self, x):
        c1 = max(0, -self.h_dot(x) / self.h(x)) + self.c_min
        c2 = max(0, -(0 + c1 * self.h_dot(x)) 
                / (self.h_dot(x) + c1 * self.h(x))) + self.c_min
        return [c1 + c2, c1 * c2]