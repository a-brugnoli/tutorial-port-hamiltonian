from scipy.sparse.linalg import spsolve
import sympy

def solve_bcs(A, b, bcs : dict = {}):
    if bcs:
        try:
            assert all(key <= A.shape[0] for key in bcs.keys())
        except AssertionError:
            raise ValueError("Boundary conditions are out of range")

            
    return spsolve(A, b)


def derive_expression(exp):
    t = sympy.symbols('t')
    exp_simpy = exp(t)
    dexp_dt = sympy.diff(exp_simpy, t)
    return sympy.lambdify(t, dexp_dt)