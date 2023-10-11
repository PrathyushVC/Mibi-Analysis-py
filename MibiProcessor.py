import os
import numpy as np
import pandas as pd
from tifffile import TiffFile
from MibiLoader import MibiLoader
import matplotlib.pyplot as plt

save_directory=r'D:\MIBI-TOFF\Data'
MibiLoader(root=None, expressiontypes=None, grps=None, T_path=None,save_directory=r'D:\MIBI-TOFF\Data')

data_catch=np.load(r'D:\MIBI-TOFF\Data\FOV1_G3_CD4.npz')
print(data_catch.files)

plt.figure()
im = plt.imshow(data_catch['imageData'], 'gray')
plt.title('ImageData')


plt.figure()
segshow = plt.imshow(data_catch['segmentation'], 'gray')
plt.title('Segmentation')

plt.figure()
clusterdseg=plt.imshow(data_catch['clustered_seg'], 'gray')
plt.title('Clustered_Seg')
plt.show()