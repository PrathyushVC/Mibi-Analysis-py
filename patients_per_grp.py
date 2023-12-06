#Script to find patient groups in Data Summary
import os,cv2
import numpy as np
import pandas as pd

def count_patients_and_FOV(df=None,name_drop=None):
    '''
    Counts the number of patients and FOV Names grouped by 'Group' in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing patient information and FOV Names.

    Returns:
    - merged_results (pandas.DataFrame): A merged DataFrame with counts of patients and FOV Names for each 'Group'.

    This function groups patients and FOV Names by the 'Group' column in the input DataFrame and calculates the number of patients and the number of unique FOV Names for each group. It then merges the results into a single DataFrame for further analysis or reporting.
    '''
    #This assumes that the blanks in the table are not needed 
    if name_drop:
        mask = df['Name'].str.contains(name_drop, case=False)#This is for testing purposes
        df = df[~mask]
    df.dropna(subset=['FOV_Name'], inplace=True)


    results = df.groupby('Group')['patient number'].apply(set).reset_index()
    results['Number of Patients'] = results['patient number'].apply(len)


    results_fov=df.groupby('Group')['FOV_Name'].apply(set).reset_index()
    results_fov['Number of FOV Name']=results_fov['FOV_Name'].apply(len)
    merged_results = pd.merge(results, results_fov, on='Group')
    return merged_results

def extract_set(merged_results=None,group='control'):
    '''
    Extracts a list of patient numbers from a specific group in the merged results DataFrame.

    Parameters:
    - merged_results (pandas.DataFrame): The DataFrame containing group information.
    - group (str): The name of the group for which patient numbers will be extracted.

    Returns:
    - patient_number_list (list of int): A list of patient numbers from the specified group.

    Usage:
    Call this function to retrieve a list of patient numbers for a specific group from the merged results DataFrame.
    '''
    condition = (merged_results['Group'] ==group )
    patient_number=merged_results.loc[condition, 'patient number'].head(1).values[0]# Pull the matching GX and get the set
    patient_number_list = [int(item) for item in patient_number]
    return patient_number_list
    


path_to_csv=r"D:\MIBI-TOFF\Data_For_Amos\Data_Summary.csv"
df = pd.read_csv(path_to_csv)
print(df.columns)
print(df.head)

merged_results=count_patients_and_FOV(df=df,name_drop='prescan')
patient_list_G1=extract_set(merged_results=merged_results,group='G1')
patient_list_G2=extract_set(merged_results=merged_results,group='G2')
patient_list_G3=extract_set(merged_results=merged_results,group='G3')
patient_list_G4=extract_set(merged_results=merged_results,group='G4')

print(patient_list_G1)
print(patient_list_G2)
print(patient_list_G3)
print(patient_list_G4)
# Save the merged DataFrame to a CSV file
merged_results.to_csv("merged_results.csv", index=False)

