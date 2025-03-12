import pandas as pd

# set the path
num = 0
path = '../other_files/'
ex_fi = ['res_summary.xlsx', 'res_summary_untrained.xlsx'][num]
ex_da = pd.read_excel(path + ex_fi)
met_li = ['Auc', 'acc', 'sen', 'spe']

me_ex = []
for eye in ex_da['Eye'].unique():
    for att in ex_da['Attribute'].unique():
        for dis in ex_da['disease'].unique():
            for gz1 in ex_da['group1 sam size'].unique():
                for gz2 in ex_da['group2 sam size'].unique():
                    ex_sel = ex_da[(ex_da['Eye'] == eye) & (ex_da['Attribute'] == att) & (ex_da['disease'] == dis) & (ex_da['group1 sam size'] == gz1) & (ex_da['group2 sam size'] == gz2)]

                    for rs in ex_sel['rand str'].unique():
                        ex_gro = ex_sel[ex_sel['rand str'] == rs]
                        ex_gro = ex_gro[ex_gro['group'] != 'whole']

                        gap_met = ex_gro[met_li].max() - ex_gro[met_li].min()
                        min_met = ex_gro[met_li].min()
                        me_ex.append([eye, att, dis, gz1, gz2, rs] + list(gap_met) + list(min_met))

df = pd.DataFrame(me_ex, columns = ['Eye', 'Attribute', 'disease', 'group1 sam size', 'group2 sam size', 'rand str'] + [x + ' gap' for x in met_li] + [x + ' min' for x in met_li])
df.to_excel(path + ex_fi.split('.')[0] + '_unfairness.xlsx', index=False)