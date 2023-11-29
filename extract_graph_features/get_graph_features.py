import numpy as np
from scipy.spatial import Delaunay, Voronoi,ConvexHull,distance
from scipy.stats import iqr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

#Needs a total rework to match the old code 1-1
'''
% graphfeats    Calculates graph-based features for nuclear centroids
% located at (x,y) in the image.
%
% Necessary input:
% x,y: x and y coordinates of points that will be used for graph construction
% (i.e. nuclear centroids).

% Output Description: vfeature contains the following:

% Voronoi Features
% 1: Area Standard Deviation
% 2: Area Average
% 3: Area Minimum / Maximum
% 4: Area Disorder
% 5: Perimeter Standard Deviation
% 6: Perimeter Average
% 7: Perimeter Minimum / Maximum
% 8: Perimeter Disorder
% 9: Chord Standard Deviation
% 10: Chord Average
% 11: Chord Minimum / Maximum
% 12: Chord Disorder

% Delaunay Triangulation
% 13: Side Length Minimum / Maximum
% 14: Side Length Standard Deviation
% 15: Side Length Average
% 16: Side Length Disorder
% 17: Triangle Area Minimum / Maximum
% 18: Triangle Area Standard Deviation
% 19: Triangle Area Average
% 20: Triangle Area Disorder

% Minimum Spanning Tree
% 21: MST Edge Length Average
% 22: MST Edge Length Standard Deviation
% 23: MST Edge Length Minimum / Maximum
% 24: MST Edge Length Disorder

% Nuclear Features
% 25: Area of polygons
% 26: Number of nuclei
% 27: Density of Nuclei
% 28: Average distance to 3 Nearest Neighbors
% 29: Average distance to 5 Nearest Neighbors
% 30: Average distance to 7 Nearest Neighbors
% 31: Standard Deviation distance to 3 Nearest Neighbors
% 32: Standard Deviation distance to 5 Nearest Neighbors
% 33: Standard Deviation distance to 7 Nearest Neighbors
% 34: Disorder of distance to 3 Nearest Neighbors
% 35: Disorder of distance to 5 Nearest Neighbors
% 36: Disorder of distance to 7 Nearest Neighbors
% 37: Avg. Nearest Neighbors in a 10 Pixel Radius
% 38: Avg. Nearest Neighbors in a 20 Pixel Radius
% 39: Avg. Nearest Neighbors in a 30 Pixel Radius
% 40: Avg. Nearest Neighbors in a 40 Pixel Radius
% 41: Avg. Nearest Neighbors in a 50 Pixel Radius
% 42: Standard Deviation Nearest Neighbors in a 10 Pixel Radius
% 43: Standard Deviation Nearest Neighbors in a 20 Pixel Radius
% 44: Standard Deviation Nearest Neighbors in a 30 Pixel Radius
% 45: Standard Deviation Nearest Neighbors in a 40 Pixel Radius
% 46: Standard Deviation Nearest Neighbors in a 50 Pixel Radius
% 47: Disorder of Nearest Neighbors in a 10 Pixel Radius
% 48: Disorder of Nearest Neighbors in a 20 Pixel Radius
% 49: Disorder of Nearest Neighbors in a 30 Pixel Radius
% 50: Disorder of Nearest Neighbors in a 40 Pixel Radius
% 51: Disorder of Nearest Neighbors in a 50 Pixel Radius
'''


def eucl(crds1, crds2=None):
    if crds2 is None:
        dists = squareform(pdist(crds1))
    else:
        dists = np.sqrt(np.sum((crds1[:, np.newaxis, :] - crds2) ** 2, axis=2))
    return dists

def mstree(crds, labels=None, doplot=False):
    n, p = crds.shape

    if labels is None:
        labels = []

    if not doplot or p < 2:
        doplot = False

    if labels:
        if not isinstance(labels, str):
            labels = [str(label) for label in labels]

    totlen = 0
    edges = np.zeros((n - 1, 2), dtype=int)
    edgelen = np.zeros(n - 1)

    dist = squareform(pdist(crds))  # Pairwise euclidean distances
    highval = np.max(dist) + 1

    e1 = np.ones(n - 1, dtype=int)
    e2 = np.arange(2, n + 1)
    ed = dist[1:, 0]

    for edge in range(n - 1):
        mindist, i = np.min(ed), np.argmin(ed)
        t, u = e1[i], e2[i]
        totlen += mindist

        if t < u:
            edges[edge, :] = [t, u]
        else:
            edges[edge, :] = [u, t]

        edgelen[edge] = mindist

        if edge < n - 1:
            i = np.where(e2 == u)[0]
            e1[i] = 0
            e2[i] = 0
            ed[i] = highval

            indx = np.where(e1 > 0)[0]
            for i in indx:
                j = indx[i]
                t, v = e1[j], e2[j]
                if dist[u, v] < dist[t, v]:
                    e1[j] = u
                    ed[j] = dist[u, v]

    if doplot:
        plot_mstree(crds, edges, labels)

    return edges, edgelen, totlen

def plot_mstree(crds, edges, labels=None):
    n, p = crds.shape

    if labels:
        if not isinstance(labels, str):
            labels = [str(label) for label in labels]

    plt.plot(crds[:, 0], crds[:, 1], 'ko')  # plot points

    for i in range(n - 1):
        x = [crds[edges[i, 0] - 1, 0], crds[edges[i, 1] - 1, 0]]
        y = [crds[edges[i, 0] - 1, 1], crds[edges[i, 1] - 1, 1]]
        plt.plot(x, y, 'r-')

    if labels:
        for i, label in enumerate(labels):
            plt.text(crds[i, 0], crds[i, 1], label, fontsize=8, ha='right')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Minimum Spanning Tree')
    plt.show()

def putbnd(x, y, buffer=0.05, nocall=False):
    x = x.flatten()
    y = y.flatten()
    indx = np.isfinite(x) & np.isfinite(y)
    x = x[indx]
    y = y[indx]

    v = np.zeros(4)
    v[0] = np.min(x) - buffer * np.ptp(x)
    v[1] = np.max(x) + buffer * np.ptp(x)
    v[2] = np.min(y) - buffer * np.ptp(y)
    v[3] = np.max(y) + buffer * np.ptp(y)

    if v[1] - v[0] < np.finfo(float).eps:
        v[0] = x[0] - 1
        v[1] = x[0] + 1

    if v[3] - v[2] < np.finfo(float).eps:
        v[2] = y[0] - 1
        v[3] = y[0] + 1

    if not nocall:
        plt.axis(v)
        plt.show()

    return v

def rowtoval(x):
    r, c = x.shape
    x_min = np.min(np.min(x))
    x -= x_min
    base = np.max(np.max(x)) + 1

    vals = np.zeros(r)
    for i in range(c):
        vals += x[:, i] * base**(c - i)

    return vals, base

def sortmat(invect, *inmats):
    args = np.argsort(invect.flatten())
    outvect = np.sort(invect.flatten())
    outmats = [np.array(inmat.flatten())[args].reshape(inmat.shape) for inmat in inmats]
    return (outvect,) + tuple(outmats)

def pnpoly(x, y, xp, yp):
    c = False
    for i in range(len(xp)):
        if (((yp[i] <= y and y < yp[i - 1]) or (yp[i - 1] <= y and y < yp[i])) and
                (x < (xp[i - 1] - xp[i]) * (y - yp[i]) / (yp[i - 1] - yp[i]) + xp[i])):
            c = not c
    return c

def get_graph_features(x, y, mask=None):    
    # Check if 'mask' variable exists
    if 'mask' in locals() or 'mask' in globals():
        validHull = plt.contour(mask, levels=[0.5], colors='black').collections[0].get_paths()[0].vertices
    else:
        K = plt.contour(x, y, levels=[0.5]).collections[0].get_paths()[0]
        validHull = np.column_stack((x[K.vertices, :], y[K.vertices]))

    # Calculate the Voronoi diagram
    points = np.column_stack((x, y))
    vor = Voronoi(points)
    VX, VY = vor.vertices.T
    V, C = vor.vertices, vor.simplices

    # Get the Delaunay triangulation
    delK = np.copy(K)

# Create delConstrain
   
    DT = Delaunay(np.column_stack((x, y)))
    IO = DT.find_simplex(np.column_stack((x, y))) >= 0
    delK = np.column_stack((x[IO], y[IO]))
    delConstrain = np.column_stack((delK[:-1], delK[1:]))
    DT = Delaunay(np.column_stack((x, y)), delConstrain)
    delTri = DT.simplices

    # Get the Minimum Spanning Tree (MST)
    mst_edges, mst_edgelen, mst_totlen = mstree(np.column_stack((x, y)), doplot=False)

    # Record indices of inf and extreme values to skip these cells later
    Vnew = np.copy(V)
    Vnew = np.delete(Vnew,0,axis=0)

    banned = []
    banned = []
    for i, point in enumerate(V):
        if not np.any(np.all(point == validHull, axis=1)):  # Check if the point is inside the validHull
            banned.append(i)

    # Voronoi Diagram Features
    # Area
    c = 0
    d = 0
    e = d
    area = []
    perimdist = []
    chorddist = []

    for i in range(len(C)):
        if not np.any(np.isin(C[i], banned)):
            X = V[C[i], :]
            chord = np.zeros((2, X.shape[0]))
            chord[0, :] = X[:, 0]
            chord[1, :] = X[:, 1]

            # Calculate the chord lengths (each point to each other point)
            for ii in range(chord.shape[1]):
                for jj in range(ii + 1, chord.shape[1]):
                    chorddist.append(np.sqrt((chord[0, ii] - chord[0, jj]) ** 2 + (chord[1, ii] - chord[1, jj]) ** 2))

            # Calculate perimeter distance (each point to each nearby point)
            for ii in range(X.shape[0] - 1):
                perimdist.append(np.sqrt((X[ii, 0] - X[ii + 1, 0]) ** 2 + (X[ii, 1] - X[ii + 1, 1]) ** 2))
            perimdist.append(np.sqrt((X[-1, 0] - X[0, 0]) ** 2 + (X[-1, 1] - X[0, 1]) ** 2))

            # Calculate the area of the polygon
            area.append(np.abs(0.5 * np.sum(X[:, 0] * np.roll(X[:, 1], 1) - np.roll(X[:, 0], 1) * X[:, 1])))

    area = np.array(area)
    perimdist = np.array(perimdist)
    chorddist = np.array(chorddist)

    # Calculate features
    vfeature = np.zeros(51)
    if len(area) > 0:
        vfeature[0] = np.std(area)
        vfeature[1] = np.mean(area)
        vfeature[2] = iqr(area) / iqr(area, rng=(95, 5))
        vfeature[3] = 1 - (1 / (1 + (vfeature[0] / vfeature[1])))

        vfeature[4] = np.std(perimdist)
        vfeature[5] = np.mean(perimdist)
        vfeature[6] = iqr(perimdist) / iqr(perimdist, rng=(95, 5))
        vfeature[7] = 1 - (1 / (1 + (vfeature[4] / vfeature[5])))

        vfeature[8] = np.std(chorddist)
        vfeature[9] = np.mean(chorddist)
        vfeature[10] = iqr(chorddist) / iqr(chorddist, rng=(95, 5))
        vfeature[11] = 1 - (1 / (1 + (vfeature[8] / vfeature[9])))
    
    vfeature=np.array(vfeature)       
#Delaunay
#edge length and area

    c = 0
    d = 0
    sidelen = []
    triarea = []

    for i in range(delTri.shape[0]):
        t = np.column_stack((x[delTri[i, :]], y[delTri[i, :]]))

        side_lengths = [
            np.sqrt((t[0, 0] - t[1, 0]) ** 2 + (t[0, 1] - t[1, 1]) ** 2),
            np.sqrt((t[0, 0] - t[2, 0]) ** 2 + (t[0, 1] - t[2, 1]) ** 2),
            np.sqrt((t[1, 0] - t[2, 0]) ** 2 + (t[1, 1] - t[2, 1]) ** 2),
        ]
        
        sidelen.extend(side_lengths)
        
        dis = np.sum(side_lengths)
        
        c += 3
        triarea.append(np.abs(np.polyarea(t[:, 0], t[:, 1])))
        d += 1


    
    vfeature[12] = np.percentile(sidelen, 5) / np.percentile(sidelen, 95)
    vfeature[13] = np.std(sidelen)
    vfeature[14] = np.mean(sidelen)
    vfeature[15] = 1 - (1 / (1 + vfeature[14] / vfeature[15]))
    vfeature[16] = np.percentile(triarea, 5) / np.percentile(triarea, 95)
    vfeature[17] = np.std(triarea)
    vfeature[18] = np.mean(triarea)
    vfeature[19] = 1 - (1 / (1 + vfeature[18] / vfeature[19]))
    vfeature[20] = np.mean(mst_edgelen)
    vfeature[21] = np.std(mst_edgelen)
    vfeature[22] = np.percentile(mst_edgelen, 5) / np.percentile(mst_edgelen, 95)
    vfeature[23] = 1 - 1 / (1 + vfeature[21] / vfeature[20])

    # Nuclear Features
    # Density
    vfeature[24] = np.sum(area)
    vfeature[25] = len(C)
    vfeature[26] = vfeature[25] / vfeature[24]

    # EdgeObjects
    edge_objects = ConvexHull(x, y, 1)

    # Distance matrix
    dist_mat = np.zeros((len(x) - len(edge_objects), len(x)))
    dist_froms = np.arange(len(x))
    dist_froms[edge_objects] = []
    dist_tos = np.arange(len(x))

    for i in range(len(dist_froms)):
        for j in range(len(x)):
            dist_mat[i, j] = np.sqrt((x[dist_froms[i]] - x[j]) ** 2 + (y[dist_froms[i]] - y[j]) ** 2)

    DKNN = np.zeros((3, dist_mat.shape[0]))
    k_count = 0

    for k in [3, 5, 7]:
        for i in range(dist_mat.shape[0]):
            tmp = np.sort(dist_mat[i, :])
            DKNN[k_count, i] = np.sum(tmp[1 : k + 1])  # Skip the first element, which is the distance to itself
        k_count += 1


    # Average Distance to K-NN
    DKNN_mean = np.mean(DKNN, axis=1)
    DKNN_std = np.std(DKNN, axis=1)

    vfeature[27:29] = DKNN_mean
    vfeature[30:32] = DKNN_std

    vfeature[33] = 1 - 1 / (1 + vfeature[30] / (vfeature[27] + np.finfo(float).eps))
    vfeature[34] = 1 - 1 / (1 + vfeature[31] / (vfeature[28] + np.finfo(float).eps))
    vfeature[35] = 1 - 1 / (1 + vfeature[32] / (vfeature[29] + np.finfo(float).eps))    

# NNRR_av: Average Number of Neighbors in a Restricted Radius
# Set the number of pixels within which to search
    rcount = 0
    NNRR_av = np.zeros((5, dist_mat.shape[0]))
    NNRR_sd = np.zeros((5, dist_mat.shape[0]))
    NNRR_dis = np.zeros((5, dist_mat.shape[0]))

    for R in range(10, 60, 10):
        rcount += 1
        
        # For each point, find the number of neighbors within R pixels
        for i in range(dist_mat.shape[0]):
            NNRR_av[rcount - 1, i] = np.sum(dist_mat[i, :] <= R) - 1
        
        if np.sum(NNRR_av[rcount - 1, :]) == 0:
            eval(f'NNRR_av_{R} = 0')
            eval(f'NNRR_sd_{R} = 0')
            eval(f'NNRR_dis_{R} = 0')
        else:
            eval(f'NNRR_av_{R} = np.mean(NNRR_av[rcount - 1, :])')
            eval(f'NNRR_sd_{R} = np.std(NNRR_av[rcount - 1, :])')
            eval(f'NNRR_dis_{R} = 1 - (1 / (1 + (NNRR_sd_{R} / NNRR_av_{R})))')

    # Update vfeature array
    #These features are weirdly generated by taking the results matrix and converting it to individual variables. need t change that.    
    vfeature[36:51] = [
        NNRR_av_10, NNRR_av_20, NNRR_av_30, NNRR_av_40, NNRR_av_50,
        NNRR_sd_10, NNRR_sd_20, NNRR_sd_30, NNRR_sd_40, NNRR_sd_50,
        NNRR_dis_10, NNRR_dis_20, NNRR_dis_30, NNRR_dis_40, NNRR_dis_50
    ]    