import argparse
import os
import pickle
import numpy as np
import pandas
from sklearn.neural_network import MLPClassifier

# read pickle file
def read_pic(fi):
    with open(fi, 'rb') as fi:
        res = pickle.load(fi)
    return res

# pass parameters to the script
pa = argparse.ArgumentParser(description='manual to this script')
pa.add_argument('--fo_num', type=int, default = 0)
ar = pa.parse_args()

# excel list
ex_li = ['../data_excel/' + x for x in os.listdir('../data_excel/')]

# all features
left_fea = read_pic('../untrained_features/21015_Fundus_retinal_eye_image_left')
right_fea = read_pic('../untrained_features/21016_Fundus_retinal_eye_image_right')

# set the path
path = '../results/'
fo_li = [x[:-5] for x in os.listdir('../data_excel/')]

or_dis_li = ['Diabetes related eye disease', 'Glaucoma', 'Macular degeneration', 'Other serious eye condition']
sh_dis_li = ['dis1', 'dis2', 'dis3', 'dis4']

for fo in fo_li:
    # check

    if not os.path.isdir(path + fo + '/'):
        os.mkdir(path + fo + '/')

    sa_fi = path + fo + '/fea_res_notrained'
    '''
    if os.path.isfile(sa_fi):
        continue
    '''
    # get excel
    ex_fi = [x for x in ex_li if fo in x][0]
    ex_info = pandas.read_excel(ex_fi)

    # get all feature
    if 'left' in fo:
        all_fea = left_fea.copy()
    else:
        all_fea = right_fea.copy()

    # get image name and labels
    dis_na = or_dis_li[sh_dis_li.index(fo.split('_')[2])]
    da_li = []
    for st in ['tr', 'va', 'te']:
        da_ex_info = ex_info.copy()
        da_ex_info = da_ex_info[da_ex_info['data_division'].isin([st])]
        na_li = list(da_ex_info['im'])
        lab_li = np.array(list(da_ex_info[dis_na])).astype(np.int32)
        # get features
        fea_li = []
        for im in na_li:
            index = all_fea[0].index(im)
            fea = all_fea[3][index]
            fea_li.append(fea)
        fea_li = np.stack(fea_li, 0)
        da_li.append([na_li, lab_li, fea_li])

    # construct MLP model
    clf = MLPClassifier(max_iter=1500)
    clf.fit(da_li[0][2], da_li[0][1])
    te_l = clf.predict_proba(da_li[2][2])

    # save result
    va_res = [da_li[2][0], da_li[2][1], te_l, da_li[2][2]]

    with open(sa_fi, 'wb') as ff:
        pickle.dump(va_res, ff)
    print(fo)
