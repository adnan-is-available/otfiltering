# solutions.py

from utils import *
from algorithms import *

from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF

### Views estimation

def solution_base_mode0 ( M, K ) :
    M = to_proba ( M )
    mu, nu = marginals ( M )
    
    G, m, D = svds ( M, k = K )
    C, A = RIOT_algo ( M, mu, nu, G, D, mode = 0 )
    return C

def solution_base_mode1 ( M, K ) :
    M = to_proba ( M )
    mu, nu = marginals ( M )
    
    G, m, D = svds ( M, k = K )
    C, A = RIOT_algo ( M, mu, nu, G, D, mode = 1 )
    return C

def solution_nmf ( M, K ) :
    M = to_proba ( M )
    mu, nu = marginals ( M )
    
    model = NMF ( n_components = K, init = 'random', random_state = 0, max_iter = 1000 )
    W = model.fit_transform ( M )
    H = model.components_
    C, A = RIOT_algo ( M, mu, nu, W, H, mode = 0 )
    return C

### Ranking estimation

def solution_notes1 ( R, K ) :
    Rp = R / R.sum()
    U, V = factoUVX ( Rp, ( R != 0 ).astype ( float ), K )
    
    mu, nu = marginals ( Rp )
    C, A = RIOT_algo ( Rp, mu, nu, U, V, mode = 0 )
    return C

def solution_notes2 ( R, K ) :
    Rp = ( R >= 2.5 ).astype ( float )
    Rp = Rp / Rp.sum()
    return solution_base_mode0 ( Rp, K )
