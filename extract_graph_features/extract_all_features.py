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
    
    if 2 in select:
        print('\nExtracting Morph Features...')
        feats.append(extract_morph_feats(bounds))

    
    if 3 in select:
        print('\nExtracting CGT Features...')
        feats.append(extract_CGT_feats(bounds))

    
    if 4 in select:
        print('\nExtracting Cluster Graph Features...')
        feats.append(extract_cluster_graph_feats(bounds))



    return feats

def extract_graph_feats(bounds):
    gb_r = [bound['centroid_r'] for bound in bounds]
    gb_c = [bound['centroid_c'] for bound in bounds]

    graphfeats = get_graph_features(np.array(gb_r), np.array(gb_c))

    return graphfeats

def extract_morph_feats(bounds):
    gb_r = [bound['r'] for bound in bounds]
    gb_c = [bound['c'] for bound in bounds]

    bad_glands = []
    feats = np.zeros((len(gb_r), 25))

    for j in range(len(gb_r)):
        if len(gb_r[j]) > 4:
            feat = morph_features(np.array(gb_r[j]), np.array(gb_c[j]))
            feats[j, :] = feat
        else:
            bad_glands.append(j)

    feats = np.delete(feats, bad_glands, axis=0)

    morphfeats = np.concatenate([np.nanmean(feats), np.nanstd(feats), np.nanmedian(feats), np.percentile(feats, 5) / np.percentile(feats, 95)])

    return morphfeats

def extract_CGT_feats(bounds):
    a = 0.5
    r = 0.2
    CGTfeats, _, _, _, _, _ = extract_CGT_features(bounds, a, r)

    return CGTfeats

def extract_cluster_graph_feats(bounds):
    info = {'alpha': 0.5, 'radius': 0.2}
    alpha = info['alpha']
    r = info['radius']
    VX, VY, x, y, edges = construct_ccgs_optimized(bounds, alpha, r)
    CCGfeats, _ = cluster_graph_features_optimized(bounds, edges)

    return CCGfeats
