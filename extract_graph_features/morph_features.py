import numpy as np
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import cov
from skimage.measure import regionprops
from skimage.morphology import label
from scipy.ndimage import binary_erosion
from shapely.geometry import Polygon

def morph_features(x, y):
    # Order the coordinates
    xy = np.column_stack((x, y))
    xy = order(xy)

    # Get the centroid and area
    xc, yc, area = calculate_centroid_area(xy)
    distance = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    dist_min = np.min(distance)
    dist_max = np.max(distance)

    # Draw circles from centroid to max and min radii
    theta = np.arange(0, 2 * np.pi, 0.01)
    x_c1 = dist_max * np.cos(theta) + xc
    y_c1 = -dist_max * np.sin(theta) - yc

    x_c2 = dist_min * np.cos(theta) + xc
    y_c2 = dist_min * np.sin(theta) + yc

    # Get variance and standard deviation
    var = np.cov(distance)
    stdv = np.std(distance)

    # Get maximum area and Area Ratio
    max_area = np.pi * dist_max ** 2
    Area_Ratio = area / max_area

    # Ratio between average distance and maximum distance
    dist_mean = np.mean(distance)
    Dist_Ratio = dist_mean / dist_max

    # Normalizing distance to find the variance and std
    dists = distance / dist_max
    dists_std = np.std(dists)
    dists_var = np.cov(dists)

    # New distance ratio defined
    dratio = distratio(xy)

    # Area to perimeter ratio
    paratio, peri = periarea(xy, area)
    if paratio == np.inf:
        paratio = np.nan

    # Smoothness Metric
    D = distance
    s = len(D)
    sm = np.zeros_like(D)

    for i in range(s):
        if i == 0:
            sm[i] = abs(D[i] - ((D[i + 1] + D[s - 1]) / 2))
        elif i == s - 1:
            sm[i] = abs(D[i] - ((D[0] + D[i - 1]) / 2))
        elif 1 <= i <= s - 2:
            sm[i] = abs(D[i] - ((D[i + 1] + D[i - 1]) / 2))

    smooth = np.sum(sm)

    # Fourier Descriptors of boundary
    z = frdescpUncentered(xy)
    # First descriptor is DC component (which is translationally variant)
    if len(z) < 11:
        z = np.concatenate((z, np.full(11 - len(z), np.nan)))
    # Divide by sum(z) for scale invariance
    fd = z[1:11] / np.nansum(z)

    # Invariant moments
    B = bound2im(xy)
    phi = invmoments(B)

    # Get the fractal dimension
    frac_dim = fractal_dim(xy)

    if frac_dim == np.inf:
        frac_dim = np.nan

    # Para are all the features that are extracted.
    feats = np.concatenate([Area_Ratio, Dist_Ratio, dists_std, dists_var, dratio, paratio, smooth, phi, frac_dim, fd])

    return feats

def order(c):

    ci = np.copy(c)
    xy = np.zeros_like(ci)

    # Reverse the order of coordinates along each dimension
    for i in range(ci.shape[1]):
        xy[ci.shape[1] - 1 - i, 0] = ci[0, i]
        xy[ci.shape[1] - 1 - i, 1] = ci[1, i]

    return xy



def calculate_centroid_area(vertices):
    
    polygon = Polygon(vertices)
    centroid = polygon.centroid
    area = polygon.area

    return centroid.x, centroid.y, area

def distratio(xy):
    n = xy.shape[0]

    # Select 1% of points or a minimum of 3 points
    points = np.round(np.linspace(0, n - 1, max(int(n * 0.01), 3))).astype(int)
    
    lx = xy[points, 0]
    ly = xy[points, 1]
    lxy = np.column_stack((lx, ly))

    # For long distances
    dislong = np.sqrt(np.sum(np.diff(lxy, axis=0)**2, axis=1))

    # For smaller distances
    disshort = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))

    dl = np.sum(dislong)
    ds = np.sum(disshort)
    dratio = dl / ds

    return dratio


def bound2im(xy):
    # Replace this function with your actual implementation of bound2im
    # It's not provided in the original question
    pass

def frdescp_uncentered(s):
    
    np, nc = s.shape
    if nc != 2:
        raise ValueError('S must be of size np-by-2')
    if np % 2 != 0:
        s = np.vstack([s, s[-1, :]])
        np += 1

    s = s[:, 0] + 1j * s[:, 1]

    z = np.fft.fft(s).real

    return z

def invmoments(B):
    # Replace this function with your actual implementation of invmoments
    # It's not provided in the original question
    pass

def fractal_dim(xy):
    # fractal dimension
    # xy being the matrix containing x and y coordinates

    n = xy.shape[0]  # assuming 2 column vectors

    d = np.zeros(n // 2)

    for i in range(1, n // 2 + 1):
        j = n // i

        lx = np.zeros(j)
        ly = np.zeros(j)

        for s in range(j):
            lx[s] = xy[1 + (s - 1) * i, 0]
            ly[s] = xy[1 + (s - 1) * i, 1]

        lx[j] = xy[n - 1, 0]
        ly[j] = xy[n - 1, 1]

        # for long distances
        dislong = np.sqrt(np.diff(lx)**2 + np.diff(ly)**2)

        d[i - 1] = np.sum(dislong)
    
    # Plotting for visualization
    #plt.plot(np.log10(1 / np.arange(1, n // 2 + 1)), np.log10(d), 'o')
    #plt.xlabel('log10(1/(1:n/2))')
    #plt.ylabel('log10(d)')
    #plt.show()

    # Performing linear regression to find the slope (fractal dimension)
    eq = np.polyfit(np.log10(1 / np.arange(1, n // 2 + 1)), np.log10(d), 1)
    frac_dim = eq[0]

    return frac_dim

def periarea(xy, area):
    n = xy.shape[0]
    dist = np.zeros(n-1)

    for i in range(n-1):
        dist[i] = np.sqrt((xy[i, 0] - xy[i+1, 0])**2 + (xy[i, 1] - xy[i+1, 1])**2)

    ld = np.sqrt((xy[n-1, 0] - xy[0, 0])**2 + (xy[n-1, 1] - xy[0, 1])**2)

    peri = np.sum(dist) + ld
    paratio = (peri**2) / area

    return paratio, peri
