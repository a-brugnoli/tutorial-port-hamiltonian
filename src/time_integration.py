from scipy.sparse.linalg import spsolve
import numpy as np


def explicit_newmark(u0, v0, M, K, dt, nt):
   
    # Solution matrix
    U = np.zeros((nt+1, len(u0)))
    U[0, :] = u0

    u_old = u0
    v_old = v0
    a_old = spsolve(M, - K @ u_old)
    a_old[0] = 0
    a_old[-1] = 0
    
    # Time-stepping loop
    for n in range(nt):
        u_new = u_old + dt * v_old + 0.5 * dt**2 * a_old
        a_new = spsolve(M, - K @ u_new)
        a_new[0] = 0
        a_new[-1] = 0

        v_new = v_old + 0.5*dt*(a_old + a_new)
        
        U[n+1, :] = u_new

        u_old = u_new
        v_old = v_new
        a_old = a_new
        
    return U

def implicit_newmark(u0, v0, M, K, dt, nt, gamma=0.5, beta=0.25):
    assert beta > 0
    # Solution matrix
    U = np.zeros((nt+1, len(u0)))
    U[0, :] = u0

    u_old = u0
    v_old = v0
    a_old = spsolve(M, - K @ u_old)
    a_old[0] = 0
    a_old[-1] = 0
    
    # Effective system matrices
    A = 1/(beta * dt**2) * M + K
    
    # Time-stepping loop
    for n in range(nt):
        b = M @ (1./(beta * dt**2) * u_old  + 1./(beta*dt) * v_old)
        u_new = spsolve(A, b)

        u_new[0] = 0
        u_new[-1] = 0


        a_new = 1/(beta * dt**2) * (u_new - u_old - dt * v_old) \
              - (1-2*beta)/(2*beta) * a_old
        v_new = v_old + dt * ((1-gamma)*a_old + gamma * a_new)

        U[n+1, :] = u_new

        u_old = u_new
        v_old = v_new
        a_old = a_new
    
    return U