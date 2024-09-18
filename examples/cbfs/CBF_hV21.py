class CBF_hV21:
    def __init__(self, x_init, dvH_th, c_min, idm):
        self.dvH_th = dvH_th
        self.c_min = c_min
        self.idm = idm
        self.alpha = self._alpha(x_init)

    def h(self, x):
        return x['vR'][1] - x['vR'][0] - self.dvH_th
    def h_dot(self, x):
        return x['aR'][1] - x['aR'][0]
    
    def constraint(self, x):
        # -u1 + u2 >= -alpha * h
        return (-1, 1, 0, -self.alpha[0] * self.h(x))
    
    def _alpha(self, x):
        c1 = max(0, -self.h_dot(x) / self.h(x)) + self.c_min
        return [c1]