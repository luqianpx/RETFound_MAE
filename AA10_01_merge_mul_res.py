import pandas as pd

# Set the path and file selection
num = 1
path = '../other_files/'
ex_fi = ['res_summary.xlsx', 'res_summary_untrained.xlsx'][num]
ex_da = pd.read_excel(path + ex_fi)

# Define metrics for easy reference
metrics = ['Auc', 'acc', 'sen', 'spe']

# Group by relevant columns and calculate the mean for each group
grouped_data = ex_da[ex_da['group'] != 'whole'].groupby(
    ['Eye', 'Attribute', 'disease', 'group1 sam size', 'group2 sam size', 'group']
)[metrics].mean().reset_index()

# Save the results to a new Excel file
grouped_data.to_excel(path + ex_fi.split('.')[0] + '_merged.xlsx', index=False)
