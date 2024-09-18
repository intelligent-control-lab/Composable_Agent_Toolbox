import sympy

if __name__ == '__main__':

    w_i, w_j, r_i, r_j, s_ix, s_iy, s_iz, v_ix, v_iy, v_iz, s_jx, s_jy, s_jz, v_jx, v_jy, v_jz = \
        sympy.symbols('w_i, w_j, r_i, r_j, s_ix, s_iy, s_iz, v_ix, v_iy, v_iz, s_jx, s_jy, s_jz, v_jx, v_jy, v_jz')

    s_i = sympy.Matrix([s_ix, s_iy, s_iz])
    v_i = sympy.Matrix([v_ix, v_iy, v_iz])
    s_j = sympy.Matrix([s_jx, s_jy, s_jz])
    v_j = sympy.Matrix([v_jx, v_jy, v_jz])
    
    lamb = sympy.Matrix([(r_i + r_j)**2])
    x = (s_i + w_i * v_i) - (s_j + w_j * v_j)

    c = lamb - x.T @ x

    Dc = sympy.Matrix([c]).jacobian([w_i, w_j])
    H = sympy.matrices.dense.hessian(c, [w_i, w_j])
    
    print(f"Gradient:\n{Dc}\n")
    print(f"Hessian:\n{H}\n")
    