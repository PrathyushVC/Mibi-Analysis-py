import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import os
import tifffile as tiff
from matplotlib.lines import Line2D
import re
import csv

def data_exploration_plots(FOV_directory, ncols=3, marker_inds=[0, 3, 5], 
                           expression_types=None, paired_cell_types=None, colorArr=None):
    """
    Generates and displays exploratory plots for the given Field of View (FOV) directory.

    Args:
        FOV_directory (str): Path to the directory containing TIF images.
        ncols (int): Number of columns in the marker expression plot layout. Defaults to 3.
        marker_inds (list): Indices of markers to include in the overlay plot. Defaults to [3, 4, 7].
        expression_types (list, optional): List of marker names. If None, defaults are used.
        paired_cell_types (list, optional): Corresponding cell types. If None, defaults are used.
        colorArr (np.ndarray, optional): Array of colors for each marker. If None, they are generated automatically.

    Returns:
        None: Displays the plots directly.
    """
    if expression_types is None:
        expression_types = ['MelanA', 'Ki67', 'SOX10', 'COL1A1', 'SMA', 
                            'CD206', 'CD8', 'CD4', 'CD45', 'CD3', 'CD20', 'CD11c']
    if paired_cell_types is None:
        paired_cell_types = ['melanoma', 'proliferation', 'proliferation', 
                             'ECM', 'ECM', 'M2 Mφ', 'T-cell', 'T-cell', 
                             'T-cell', 'T-cell', 'B-cell', 'DC']
    if colorArr is None:
        colorArr = np.array([
            [0.70, 0.88, 1.0],
            [0.50, 1.0, 0.75],
            [0.80, 0.80, 0.65],
            [1.0, 0.6, 0.0],
            [0.95, 1.0, 0.25],
            [0.30, 0.85, 1.0],
            [0.95, 0.95, 0.95],
            [1.0, 0.4, 0.70],
            [0.20, 0.80, 0.20],
            [1.0, 0.3, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.8, 1.0]])

    image_data = []
    plt.ion
    # Load images
    for marker in expression_types:
        marker_filepath = os.path.join(FOV_directory, 'TIFS', f"{marker}.tif")
        try:
            marker_image = tiff.imread(marker_filepath)
            image_data.append(marker_image)
        except FileNotFoundError:
            print(f"File not found: {marker_filepath}")
            continue

    nrows = int(np.ceil(len(expression_types) / ncols))

    # Plot marker expressions with individual colors
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flatten()

    for i, (ax, data) in enumerate(zip(axes, image_data)):
        color_mapped_image = create_colored_image(data, colorArr[i])
        ax.imshow(color_mapped_image)
        ax.set_title(f"{expression_types[i]} ({paired_cell_types[i]})", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    fov_name = extract_fov(FOV_directory)
    print(fov_name)

    for i, (data, expression, cell_type) in enumerate(zip(image_data, expression_types, paired_cell_types)):
        color_mapped_image = create_colored_image(data, colorArr[i])
        plt.figure(figsize=(10, 10))
        plt.imshow(color_mapped_image)
        plt.title(f"{expression} ({cell_type})", fontsize=8)
        plt.axis('off')
        output_filepath = os.path.join(f"{fov_name}_{expression}_{cell_type}.png")
        plt.savefig(output_filepath, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off') 

    base_image = image_data[marker_inds[0]]
    base_color = colorArr[marker_inds[0]]
    base_colored_image = create_colored_image(base_image, base_color)
    ax.imshow(base_colored_image, alpha=(base_image > 0).astype(float) * 0.6)  
    for idx in marker_inds[1:]:
        I = image_data[idx]
        color = colorArr[idx]
        colored_image = create_colored_image(I, color)


        alpha = (I > 0).astype(float) * 0.6
        ax.imshow(colored_image, alpha=alpha)

    plt.show()

def create_colored_image(image, color):
    """
    Creates a colored version of the grayscale image, mapping it from black to the specified color.

    Args:
        image (np.ndarray): Grayscale image array.
        color (np.ndarray): RGB color for the marker.

    Returns:
        np.ndarray: RGB image with the applied color.
    """
    normalized_image = image / np.max(image) if np.max(image) > 0 else image
    colored_image = np.zeros((*image.shape, 3), dtype=float)
    
    for i in range(3):
        colored_image[..., i] = normalized_image * color[i]
    
    return colored_image

def imlegend(marker_inds, colorArr, labelsArr, cell_types):
    """
    Displays a legend for the overlaid markers.

    Args:
        marker_inds (list): Indices of the markers to be included in the legend.
        colorArr (np.ndarray): Array of RGB colors for each marker.
        labelsArr (list): List of marker labels.
        cell_types (list): List of corresponding cell types.

    Returns:
        None: Displays the legend alongside the plot.
    """
    handles = [Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colorArr[idx], markersize=10, 
                      label=f"{labelsArr[idx]} ({cell_types[idx]})") 
               for idx in marker_inds]
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

def extract_fov(path):
    match=re.search(r'F0V\d+',path)
    return match.group(0) if match else 'Unknown'

def getTumorCount(filePath):

    table=[]
    patient={}

    '''
    Each table entry looks like:
    [Patient #, Group#, Tumor Cell Count#, FOVS]
    
    Below code is only for group 3 and group 4.
    
    '''
    validGroups=["G3","G4"]
    with open(filePath,mode='r') as file:
        cellClassifications=csv.reader(file)

        for row in cellClassifications:
            if (row[0] == ''):
                continue
            patientNum, GroupNum, cellType,FOV= int(row[-1]), row[-2], row[-7],row[-8]
            if GroupNum in validGroups:

                if (patientNum not in patient):
                    patient[patientNum]=[GroupNum,0,set()] 
                    
                if (cellType=='tumor'):
                    patient[patientNum][1]+=1 
                    patient[patientNum][2].add(FOV)
       
        for p in patient:
            table.append([p,patient[p][0],patient[p][1],", ".join(patient[p][2])])
        table=sorted(table)
        with open('Tumor_Count_With_FOV.csv', 'w') as fp:
            header=["Patient", "Group", "Tumor Cell Count", "FOVs"]
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(header)
            for row in table:
                writer.writerow(row)
        

        return
    

def getCD4TCount(filePath):

    table=[]
    patient={}

    '''
    Each table entry looks like:
    [Patient #, Group#, CD4T Cell Count#, FOVS]
    
    Below code is only for group 1 and group 2.
    
    '''
    validGroups=["G1","G2"]
    with open(filePath,mode='r') as file:
        cellClassifications=csv.reader(file)

        for row in cellClassifications:
            if (row[0] == ''):
                continue
            patientNum, GroupNum, cellType,FOV= int(row[-1]), row[-2], row[-7],row[-8]
            if GroupNum in validGroups:

                if (patientNum not in patient):
                    patient[patientNum]=[GroupNum,0,set()] 
                    
                if (cellType=='CD4 T cell'):
                    patient[patientNum][1]+=1 
                    patient[patientNum][2].add(FOV)
       
        for p in patient:
            table.append([p,patient[p][0],patient[p][1],", ".join(patient[p][2])])
        table=sorted(table)
        with open('CD4_T_Count_With_FOV.csv', 'w') as fp:
            header=["Patient", "Group", "Tumor Cell Count", "FOVs"]
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(header)
            for row in table:
                writer.writerow(row)
        

        return
        

if __name__ == "__main__":
    FOV_directory = 'D:/MIBI-TOFF/Data_For_Amos/FOV176'  # Adjust your directory path
    data_exploration_plots(FOV_directory)
