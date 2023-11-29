import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask2bounds(img, is_object_black=False):
    # Threshold to get rid of bad image compression in jpg
    pres = np.max(img)
    img[img > pres / 2] = pres
    img[img < pres / 2] = 0

    if is_object_black:
        mask = np.logical_not(img).astype(np.uint8)  # object is black
    else:
        mask = img.astype(np.uint8)  # object is white

    # Fill holes in the mask
    mask = cv2.fillPoly(mask, [mask], 1)

    # Find contours and centroids using OpenCV
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounds = []
    for contour in contours:
        r = contour[:, 0, 1].tolist()
        c = contour[:, 0, 0].tolist()

        centroid = np.mean(contour, axis=0)[0]
        centroid_r = centroid[1]
        centroid_c = centroid[0]

        bounds.append({
            'r': r,
            'c': c,
            'centroid_r': centroid_r,
            'centroid_c': centroid_c
        })

    return bounds
def visualize_bounds(img, bounds):
    plt.imshow(img, cmap='gray')
    
    for i in range(len(bounds)):
        plt.plot(bounds[i]['c'] + [bounds[i]['c'][0]], bounds[i]['r'] + [bounds[i]['r'][0]], 'b-')
        plt.plot(bounds[i]['centroid_c'], bounds[i]['centroid_r'], 'r.')

    plt.show()