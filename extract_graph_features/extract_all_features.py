import numpy as np
import get_graph_features
def extract_all_features(bounds, img=None, select=None, mask=None):
    

    if select is None:
        select = [1, 2]
    
    if mask is None:
        mask = []

    feats = []

    if 1 in select:
        print('\nExtracting Graph Features...')
        feats.append(extract_graph_feats(bounds))

    return feats

def extract_graph_feats(bounds):
    gb_r = [bound['centroid_r'] for bound in bounds]
    gb_c = [bound['centroid_c'] for bound in bounds]

    graphfeats = get_graph_features(np.array(gb_r), np.array(gb_c))

    return graphfeats
