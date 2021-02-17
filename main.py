# main.py

from utils import *
from loaddb import *
from algorithms import *
from solutions import *

M = load_database ( 'ml-1m' )
M_train = M[:5000,:]
M_test = M[5000:,:]
M = to_proba ( M_train )
mask = np.random.rand ( M.shape[1] ) < 0.5

k = 30
l = 5.

score = 0.

Cp = solution_base_mode1 ( M_train, k )
D = np.array ( [ [ np.square ( Cp[:,j1] - Cp[:,j2] ).sum() for j1 in range(Cp.shape[1]) ] for j2 in range(Cp.shape[1] ) ] )

for x in range ( 500 ) :
    nu_best = NU_algo ( M_test[x,:] * mask, D, l = l )
    nu_sort = sorted ( range(M.shape[1]), key = lambda j : - nu_best[j] )

    Nb = np.sum ( ( M_test[x,:] != 0 ) )
    if Nb != 0 :
        score += np.sum ([1. if M_test[x,nu_sort[y]] > 0 else 0. for y in range(Nb) ]) / Nb

print ( 'k = {}, l = {}, score = {:.2f}%'.format ( k, l, score / 5. ) )