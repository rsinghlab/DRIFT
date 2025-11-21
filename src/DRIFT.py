import numpy as np
import networkx as nx
from scipy.linalg import expm
import ot
from scipy.sparse import csc_matrix

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
            
    threshold = np.partition(distance_matrix.flatten(), (n_neighbors*n_spot))[(n_neighbors*n_spot) - 1]
    interaction = (distance_matrix<= threshold).astype('uint8')
    np.fill_diagonal(interaction, 0)
         
    adata.obsm['graph_neigh'] = interaction
    
    
def DRIFT(adata, t = 2):
    
    construct_interaction(adata)

    G = nx.from_numpy_array(adata.obsm['graph_neigh'])
    L = nx.laplacian_matrix(G).toarray()

    x = adata.X
    if not isinstance(x, np.ndarray):
        x = x.todense()


    heat_kernel = expm(-t * L)  # shape: (5, 5) #Heat kernel
    xt = heat_kernel @ x

    xt = csc_matrix(xt)
    adata.X = xt
    
    return adata