import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths configuration
num = 1
path = '../other_files/'
fi_pa = '../Figure_unfairness/'
ex_fi = ['res_summary_unfairness.xlsx', 'res_summary_untrained_unfairness.xlsx'][num]
ex_na = ['finetune', 'feaextra'][num]
ex_da = pd.read_excel(path + ex_fi)

# Variables
at_li = ['age', 'sex']
hue_order = [['< 60', '>= 60'], ['Female', 'Male']]
dis_li = ['dis1', 'dis2', 'dis3', 'dis4']
dis_order = ['Diabetes', 'Glaucoma', 'MD', 'Other']
met_li1 = ['Auc gap', 'acc gap', 'sen gap', 'spe gap']
met_li2 = ['Auc min', 'acc min', 'sen min', 'spe min']
met_na_li = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']

# Function for filtering and renaming data
def prepare_group_data(ex_sel, group_size_column, hue_value):
    group_data = ex_sel[ex_sel[group_size_column] == 100]
    group_data = group_data.rename(columns={'group2 sam size': 'group sam size'} if group_size_column == 'group1 sam size' else {'group1 sam size': 'group sam size'})
    group_data = group_data.drop(columns=[group_size_column])
    group_data.insert(4, "Type", [hue_value for x in range(group_data.shape[0])], True)
    return group_data

# Loop through all combinations of Eye, Attribute, and Disease
for eye in ex_da['Eye'].unique():
    for att in ex_da['Attribute'].unique():
        for dis in ex_da['disease'].unique():

            # Filter data
            ex_sel = ex_da[(ex_da['Eye'] == eye) & (ex_da['Attribute'] == att) & (ex_da['disease'] == dis)]

            # Prepare group 1 and group 2 data
            ex_gr1 = prepare_group_data(ex_sel, 'group1 sam size', hue_order[at_li.index(att)][0] + '=100')
            ex_gr2 = prepare_group_data(ex_sel, 'group2 sam size', hue_order[at_li.index(att)][1] + '=100')

            # Combine group 1 and group 2
            ex_gr = pd.concat([ex_gr1, ex_gr2])

            # Plotting
            fig, axs = plt.subplots(4, 2, figsize=(12, 9))
            for i in range(4):
                sns.lineplot(ex_gr, x='group sam size', y=met_li1[i], hue='Type', err_style="bars", ax=axs[i, 0])
                sns.lineplot(ex_gr, x='group sam size', y=met_li2[i], hue='Type', err_style="bars", ax=axs[i, 1])

            plt.tight_layout()
            plt.savefig(fi_pa + ex_na + '_' + eye + '_' + att + '_' + dis + '.svg')
            plt.savefig(fi_pa + ex_na + '_' + eye + '_' + att + '_' + dis + '.jpg')
            plt.show()
            plt.close()
