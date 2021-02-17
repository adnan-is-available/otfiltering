# algorithms.py

import numpy as np
import scipy.optimize
from utils import *

### Algorithms of Optimal Transport  

# our implementation of OT (using scipy.optimize)
def OT_algo ( mu, nu, C ) :
    n, m = mu.shape[0], nu.shape[0]
    c = C.reshape ( ( n*m, ) )
    A1 = np.array ( [ [ ij//m==i for ij in range(n*m) ] for i in range(n) ], dtype = np.double )
    A2 = np.array ( [ [ ij%m==j for ij in range(n*m) ] for j in range(m) ], dtype = np.double )
    
    #~ si on veut se ramener uniquement à des inégalités : 
    # res = scipy.optimize.linprog ( c, A_ub = np.concatenate ( ( A1, -A1, A2, -A2 ) ), b_ub = np.concatenate ( ( mu, -mu, nu, -nu ) ), method = 'revised simplex' )
    res = scipy.optimize.linprog ( c, A_eq = np.concatenate ( ( A1, A2 ) )[:-1], b_eq = np.concatenate ( ( mu, nu ) )[:-1], method = 'revised simplex' )
    return res['x'].reshape ( C.shape )

# our implementation of Sinkhorn-Knopp
def SK_algo ( mu, nu, C, l = 1., N = 15 ) :
    """ l := poids de la régularisation,
        N := nombre d'itérations
    """
    K = np.exp ( - C * l )
    a = np.ones ( mu.shape )
    for i in range(N) :
        b = nu / ( K.T @ a )
        a = mu / ( K @ b )
    return np.diag ( a ) @ K @ np.diag ( b )

def SK2_algo ( mu, nu, C, l = 1., N = 15 ) :
    return (SK_algo(mu,nu,C,l,N)*C).sum()

# our implementation of Regularized Inverse Optimal Transport
def RIOT_algo ( pi, mu, nu, G, D, mode = 0 ) :
    """ pi := plan source (n x m),
        mu := marginale (n),
        nu := marginale (m),
        G := réduction dimension (n x p),
        D := réduction dimension (q x m),
        mode := 0 -> matrice de rang faible,
             |  1 -> matrice creuse
             |  2 -> autre (non codé)
    """
    
    eps = 1e-2
    gamma = 1e-5
    n, m = mu.shape[0], nu.shape[0]
    
    Gmp = np.linalg.pinv ( G )
    Dmp = np.linalg.pinv ( D )
    
    c = np.random.rand ( n, m )
    u = np.exp ( np.random.rand ( n ) / eps )
    v = np.exp ( np.random.rand ( m ) / eps )
    
    for i in range ( 10 ) :
        K = np.exp ( -c/eps )
        u = mu / ( K @ v )
        v = nu / ( K.T @ u )
        K = pi / ( np.outer ( u, v ) ) + 1e-9
        cX = - eps * Gmp @ np.log ( K ) @ Dmp
        if mode == 0 :
            U, S, V = np.linalg.svd ( cX, full_matrices = False )
            a = U @ np.maximum ( np.diag ( S - gamma ), 0 ) @ V
        elif mode == 1 :
            a = np.sign ( cX ) * np.maximum ( np.abs ( cX ) - gamma, 0 )
        else : 
            print ( "RIOT mode not supported..." )
            exit ( 1 )
        c = G @ a @ D
    return c, a

# nu optimal pour W(mu,nu,C) régularisé
def NU_algo ( mu, C, l = 1. ) :
    K = np.exp ( - C * l )
    u = mu / K.sum(axis=1)
    return K.T @ u

### Algorithmes de factorisation par DG

# factorisation M = UV par descente de gradient
def factoUV ( M, k ) :
    U = np.random.rand ( M.shape[0], k )
    V = np.random.rand ( k, M.shape[1] )
    
    dt = 1e-5
    
    dU, dV = None, None
    i = 0
    
    while (dU is None) or np.square(M-U@V).sum() > 1e-4 :
        i += 1
        dU = - 2 * ( M - U@V ) @ V.T
        dV = - 2 * U.T @ ( M - U@V )
        U -= dt * dU
        V -= dt * dV
    return ( U, V )

# factorisation M = UV avec V fixé
def factoU ( M, V ) :
    U = np.random.rand ( M.shape[0], V.shape[0] )
    
    dt = 1e-2
    dU = None
    
    while (dU is None) or np.square(dU).sum() > 1. :
        dU = - 2 * ( M - U@V ) @ V.T
        print ( np.square(dU).sum() )
        U -= dt * dU
    return ( U, V )

# factorisation M*X = UV*X (X est un masque, * le produit élément par élément)
def factoUVX ( M, X, k ) :
    U = np.random.rand ( M.shape[0], k )
    V = np.random.rand ( k, M.shape[1] )
    
    dt = 1e-5
    
    dU, dV = None, None
    i = 0
    
    while (dU is None) or np.square(X*(M-U@V)).sum() > 1e-3 :
        i += 1
        dU = - 2 * ( X * ( M - U@V ) ) @ V.T
        dV = - 2 * U.T @ ( X * ( M - U@V ) )
        U -= dt * dU
        V -= dt * dV
    return ( U, V )

