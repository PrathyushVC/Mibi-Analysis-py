import csv


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
        


if __name__=="__main__":
    getCD4TCount("cleaned_expression_with_both_classification_prob_spatial_27_09_23.csv")
        
