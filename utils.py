# utils.py

import numpy as np

def to_proba ( u ) :
    up = ( u != 0 ).astype ( float ) 
    return up / up.sum()

def to_prob ( u ) : return u / u.sum()

def marginals ( pi ) :
    return ( pi.sum ( axis = 1 ), pi.sum ( axis = 0 ) )

def top_25 ( MTrain, My, cache, cost_function ) :
    y_min = sorted ( range(len(My)), key = lambda y : cost_function ( MTrain[:,y], to_proba ( My * cache ) ) )
    return sum ( 1. if My[y]>0 else 0. for y in y_min[:25] ) / 25.

def top_k ( MTrain, My, cache, cost_function ) :
    Nb = len([1. for y in range(len(My)) if My[y]>0.])
    y_min = sorted ( range(len(My)), key = lambda y : cost_function ( MTrain[:,y], to_proba ( My * cache ) ) )
    if Nb > 0 : return sum ( 1. if My[y]>0 else 0. for y in y_min[:Nb] ) / Nb
    else : return 0.


def nu_to_notes ( nu ) :
    if nu[len(nu)-1] > 0 :
        return 2.5 * nu / nu[len(nu)-1]
    return np.ones ( nu.shape ) / len(nu)

def rmse ( RTrain, Ry, cache, res_function ) :
    Ry_est = nu_to_notes ( res_function ( Ry * cache ) )
    visible = Ry > 0
    return np.sqrt ( np.square(visible*(Ry-Ry_est)).sum() / len(Ry) )