from scipy.sparse.linalg import spsolve
import sympy as sym

def solve_bcs(A, b, bcs : dict = {}):
    if bcs:
        try:
            assert all(key <= A.shape[0] for key in bcs.keys())
        except AssertionError:
            raise ValueError("Boundary conditions are out of range")

            
    return spsolve(A, b)


def derive_expression(exp):
    x = sym.symbols('x')
    dexp_dx = sym.diff(exp(x), x)
    return sym.lambdify(x, dexp_dx)

