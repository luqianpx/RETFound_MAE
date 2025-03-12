import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the path and variables
num = 0
path = '../other_files/'
fi_pa = '../Figures/'
ex_fi = ['res_summary.xlsx', 'res_summary_untrained.xlsx'][num]
ex_na = ['finetune', 'feaextra'][num]
ex_da = pd.read_excel(path + ex_fi)

at_li = ['age', 'sex']
hue_order = [['< 60', '>= 60'], ['Female', 'Male']]
dis_li = ['dis1', 'dis2', 'dis3', 'dis4']
dis_order = ['Diabetes', 'Glaucoma', 'MD', 'Other']
met_li = ['Auc', 'acc', 'sen', 'spe']
met_na_li = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']

# Function for grouping and renaming
def prepare_group_data(ex_sel, group_size_column, hue_value):
    group_data = ex_sel[ex_sel[group_size_column] == 100]
    group_data = group_data.rename(columns={'group2 sam size': 'group sam size'} if group_size_column == 'group1 sam size' else {'group1 sam size': 'group sam size'})
    group_data = group_data.drop(columns=[group_size_column])
    group_data['group'] = group_data['group'].apply(lambda x: f"{hue_value}=100, {x}")
    return group_data

# Loop through all combinations of Eye, Attribute, and Disease
for eye in ex_da['Eye'].unique():
    for att in ex_da['Attribute'].unique():
        for dis in ex_da['disease'].unique():

            # Filter data
            ex_sel = ex_da[(ex_da['Eye'] == eye) & (ex_da['Attribute'] == att) & (ex_da['disease'] == dis)]
            ex_sel = ex_sel[ex_sel['group'] != 'whole']

            # Prepare group 1 and group 2 data
            group_1_data = prepare_group_data(ex_sel, 'group1 sam size', hue_order[at_li.index(att)][0])
            group_2_data = prepare_group_data(ex_sel, 'group2 sam size', hue_order[at_li.index(att)][1])

            # Combine group 1 and group 2
            combined_data = pd.concat([group_1_data, group_2_data])

            # Plotting
            fig, axs = plt.subplots(4, 1, figsize=(7, 9))
            hue_or = [f"{hue_order[at_li.index(att)][0]}=100, {x}" for x in hue_order[at_li.index(att)]] + [f"{hue_order[at_li.index(att)][1]}=100, {x}" for x in hue_order[at_li.index(att)]]

            for i, met in enumerate(met_li):
                sns.lineplot(combined_data, x='group sam size', y=met, hue='group', hue_order=hue_or, err_style="bars", ax=axs[i])
                axs[i].set_ylim(combined_data[met].min()-0.05, combined_data[met].max()+0.05)
                axs[i].set_ylabel(met_na_li[met_li.index(met)])
                axs[i].set_xl
