# loaddb.py

import numpy as np
import pandas as pd

def load_database ( db_name, ext = None, sep = None, names = None, N = 1000 ) :
    base_ext = { 'ml-1m' : 'dat', 'ml-25m' : 'csv' }
    base_sep = { 'ml-1m' : '::', 'ml-25m' : ',' }
    base_names = { 'ml-1m' : ['userId', 'movieId', 'rating', 'timestamp'], 'ml-25m' : None }
    if sep == None and db_name in base_sep : sep = base_sep [db_name]
    if ext == None and db_name in base_ext : ext = base_ext [db_name]
    if names is None and db_name in base_names : names = base_names [db_name]
    print ( db_name, ext, sep )
    
    ratings = pd.read_table ( db_name + '/ratings.' + ext, sep = sep, names = names, engine = 'python', chunksize = 10000 )
    ratings = ratings.get_chunk ( 1_000_000 )
    
    ratings_count = ratings.groupby(by='movieId', as_index=True).size()
    top_ratings = ratings_count[ratings_count>=N]
    ratings = ratings[ratings.movieId.isin(top_ratings.index)]
    
    R_df = ratings.pivot ( index = 'userId', columns = 'movieId', values = 'rating' )
    return np.nan_to_num ( R_df.to_numpy() )