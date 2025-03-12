import os
import shutil
import numpy as np

# set the path
or_pa = 'E:/Project19_transfer_unfair/Data/fundu_imgs/'
so_pa = 'E:/Project19_transfer_unfair/Model/model01_RETFound/RETFound_MAE-main/dataset/GDPH/'

fo_li = os.listdir(or_pa)
for fo in fo_li:
    n = int(fo.split('-')[0])
    if n <= 45:
        cfo = 'class_A'
    else:
        cfo = 'class_B'

    rf = np.random.random()
    if rf <=0.7:
        pfo = 'train'
    elif rf > 0.7 and rf <= 0.85:
        pfo = 'test'
    else:
        pfo = 'val'
    sofu_fo = so_pa + pfo + '/' + cfo + '/'
    orfu_fo = or_pa + fo + '/'
    fo_dir = os.listdir(orfu_fo)
    for fi in fo_dir:
        fu_fi = orfu_fo + fi
        shutil.copyfile(fu_fi, sofu_fo + fi)
