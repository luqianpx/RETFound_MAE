import pickle
import pandas


def get_li(attr_str):
    if attr_str == 'age':
        return ['< 60', '>= 60']

    if attr_str == 'sex':
        return ['Female', 'Male']

    if attr_str == 'race':
        return ['White', 'Asian', 'Black']

    if attr_str == 'faminc':
        return ['Less than 18,000', '18,000 to 30,999', '31,000 to 51,999', '52,000 to 100,000', 'Greater than 100,000']

with open('../other_files/res_summary', 'rb') as fi:
    all_res = pickle.load(fi)

ex_info = []
for li in all_res:
    st_li = li[0].split('_')
    met = li[1]['best']
    ex_info.append(st_li + ['whole'] + list(met[0]))
    atr_li = get_li(st_li[1])
    for i in range(met[1].shape[0]):
        ex_info.append(st_li + [atr_li[i]] + list(met[1][i, :]))

df = pandas.DataFrame(ex_info, columns = ['Eye', 'Attribute', 'disease', 'data type', 'group1 sam size', 'group2 sam size', 'rand str', 'group', 'Auc', 'acc', 'sen', 'spe'])
df.to_excel('../other_files/res_summary.xlsx', index=False)