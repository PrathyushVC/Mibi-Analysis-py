# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# get paths to summary data spreadsheet, cell-wise expression spreadsheet, and the actual data files (both the larger dataset and the small initial test dataset; note that the large dataset does not contain the files from the small test dataset)
root = r'C:\Users\sofiy\OneDrive - Case Western Reserve University\melanoma'
cell_wise_expression_filepath = os.path.join(root,'cleaned_expression_with_both_classification_prob_spatial_27_09_23.csv')
data_summary_filepath = os.path.join(root,'Data_Summary.xlsx')
small_dataset_path = r'C:\Users\sofiy\OneDrive - Case Western Reserve University\Documents\Research Rotations\Invent Lab\1. MIBI-TOF Project\Data'
 

# read in data from cell-wise spreadsheet
cell_wise_data = pd.read_csv(cell_wise_expression_filepath)
# extract unique FOV names
cell_wise_FOV_arr = np.sort(pd.unique(cell_wise_data['fov']))


# read in data from summary data spreadsheet
summary_data = pd.read_excel(data_summary_filepath)
# make formatting of FOV names consistent with the cell-wise spreadsheet and data file names (remove the '_' before the numbers and insert a '_' before any 'G's)
summary_data['FOV_Name'] = summary_data['FOV_Name'].str.replace('_', '')
summary_data['FOV_Name'] = summary_data['FOV_Name'].str.replace('G', '_G')
#drop duplicate rows (rows assigned to the same FOV, patient, and group)
summary_data = (summary_data.drop_duplicates('FOV_Name')).dropna()
#sort values by FOV name
summary_data = summary_data.sort_values('FOV_Name')
# extract unique FOV names
summary_data_FOV_arr = np.sort(pd.unique(summary_data['FOV_Name'].dropna()))


# read in the folder names from the actual data we have and save the name of each folder beginning with 'FOV'
# the "large" dataset, containing all FOVs aside from the ones encompassed in the small initial test dataset
dirlist = os.listdir(root)
actual_data_FOV_arr = np.sort(np.array([file for file in os.listdir(root) if file.startswith('FOV')]))

# small initial test dataset
for subdir, dirs, files in os.walk(small_dataset_path):
    for directory in dirs:
        if directory.startswith('FOV'):
            actual_data_FOV_arr = np.append(actual_data_FOV_arr,directory)

actual_data_FOV_arr = np.sort(actual_data_FOV_arr)




# find the complete set of FOVs we're supposed to have (the union of FOVs from the cell-wise data, data summary spreadhseet, and the actual data files)
all_FOVs_arr = np.array(list(set(summary_data_FOV_arr).union(set(actual_data_FOV_arr)).union(set(cell_wise_FOV_arr))))
all_FOVs_arr = np.sort(all_FOVs_arr)



# create a dataframe containing the complete list of FOVs and populate the dataframe with info about which file(s) each FOVs come from
df_FOVs = pd.DataFrame(columns = ['FOV Name',
                                  'Filename',
                                  'Patient Number',
                                  'Group',
                                  'Summary Data Spreadsheet',
                                  'Cell-Wise Data Spreadsheet',
                                  'Actual Data Files'])

df_FOVs['FOV Name'] = all_FOVs_arr

# find the rows in df_FOVs that contain FOVs that came from the summary data spreadsheet 
inds_summary_data_FOVs = np.where(np.isin(all_FOVs_arr,summary_data_FOV_arr))[0]
# for the column corresponding to the summary data spreadsheet, put an 'X' the rows in the dataframe corresponding to the FOVs that came from the summary data spreadsheet
df_FOVs.loc[inds_summary_data_FOVs,'Summary Data Spreadsheet'] = 'X'
# also, copy the patient number, group assignment, and filename from the summary data spreadsheet into the same row
df_FOVs.loc[inds_summary_data_FOVs,['Filename','Patient Number','Group']] = np.array(summary_data[['Name','patient number', 'Group']])


# find the rows in df_FOVs that contain FOVs that came from the cell-wise data spreadsheet
inds_cell_wise_FOVs = np.where(np.isin(all_FOVs_arr,cell_wise_FOV_arr))[0]
# for the column corresponding to the cell-wise data spreadsheet, put an 'X' the rows in the dataframe corresponding to the FOVs that came from the cell-wise data spreadsheet
df_FOVs.loc[inds_cell_wise_FOVs,'Cell-Wise Data Spreadsheet'] = 'X'


# find the rows in df_FOVs that contain FOVs that came from the actual data files
inds_actual_data_FOVs = np.where(np.isin(all_FOVs_arr,actual_data_FOV_arr))[0]
# for the column corresponding to the actual data files, put an 'X' the rows in the dataframe corresponding to the FOVs that came from the cell-wise data spreadsheet
df_FOVs.loc[inds_actual_data_FOVs,'Actual Data Files'] = 'X'

# save as CSV file
df_FOVs.to_csv(os.path.join(root,'missing_FOVs.csv'), index = False)
