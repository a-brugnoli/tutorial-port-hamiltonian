import numpy as np
from scipy.sparse import  lil_matrix


def mass_matrix_lagrange(coordinates, parameter=1):

    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    M = lil_matrix((n_nodes, n_nodes))

    # Numerical quadrature (2-point Gauss quadrature)
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    gauss_weights = [1.0, 1.0]

    for i in range(n_elements):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  
        midpoint_element = (x_left + x_right) / 2
        for gp, gw in zip(gauss_points, gauss_weights):
            x_gp = midpoint_element + gp * element_length / 2
            N_lagrange = 1/element_length * np.array([(x_right - x_gp) , (x_gp - x_left)])
            M[i:i+2, i:i+2] += parameter * gw * element_length / 2 * np.outer(N_lagrange, \
                                                                              N_lagrange)

    M = M.tocsr()

    return M




def discrete_gradient(coordinates):
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1

    D = lil_matrix((n_elements, n_nodes))

    for i in range(n_elements):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  
        dN = 1/element_length * np.array([-1, 1])

        D[i, i:i+2] += element_length *  dN

    D = D.tocsr()

    return D

def stiffness_matrix(coordinates, stiffness=1):

    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1

    K = lil_matrix((n_nodes, n_nodes))

    for i in range(n_elements):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  # Element length
        dN = 1/element_length * np.array([-1, 1])
        K[i:i+2, i:i+2] += stiffness * element_length  * np.outer(dN, dN)

    K = K.tocsr()

    return K