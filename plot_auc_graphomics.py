import pandas as pd
import matplotlib.pyplot as plt
import ast


def plot_AUCS(data,title,ylim=(0 ,1)):
    plt.ion()

    base_label = data.iloc[0, 0]
    x_tick_labels = [f"{base_label} ({col})" for col in data.iloc[2, 0:]]

    means = data.iloc[6, 0:].apply(lambda x: ast.literal_eval(x)[0])
    lower_ranges = data.iloc[6, 0:].apply(lambda x: ast.literal_eval(x)[1])
    upper_ranges = data.iloc[6, 0:].apply(lambda x: ast.literal_eval(x)[2])


    errors = [means - lower_ranges, upper_ranges - means]

    plt.figure(figsize=(10, 10))
    plt.bar(x_tick_labels, means, yerr=errors, capsize=5,linewidth=2)
    plt.axhline(y=0.6, color='r', linestyle='-')
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Features',fontsize=15)
    plt.ylabel('Values',fontsize=15)
    plt.title(f"6 fold CV with range bars- {title}",fontsize=25)
    plt.ylim(ylim)
    plt.tight_layout()

    for i, value in enumerate(means):
        plt.text(i, 0.30, round(value, 2), ha='center', va='bottom', color='white', fontsize=25)
    
    plt.savefig(f'auc_graphomics_plot_{title}.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    plt.ioff()
    file_path = r'C:\Users\chirr\OneDrive - Case Western Reserve University\mibi.xlsx'
    sheet=r'AUC_ResultsG1vG2'
    data_G1vG2 = pd.read_excel(file_path,sheet_name=sheet, header=None)
    plot_AUCS(data_G1vG2,'G1G2',ylim=(0.35 ,1))

    sheet=r'AUC_ResultsG3vG4'
    data_G3vG4 = pd.read_excel(file_path,sheet_name=sheet, header=None)
    plot_AUCS(data_G3vG4,'G3G4',ylim=(0.35 ,1))

    sheet=r'AUC_ResultsMvNM'
    data_MvNM = pd.read_excel(file_path,sheet_name=sheet, header=None)
    plot_AUCS(data_MvNM,'MvNM',ylim=(0.0 ,1))



