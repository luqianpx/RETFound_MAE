import numpy as np
from sklearn import metrics



# calculate metric
def cal_met_without_opt(pred, lab):
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(lab, pred[:, 1], pos_label=1)
    Auc = metrics.auc(fpr, tpr)

    # acc, sen, spe, and best cutoff value
    pl, gr = pred[:, 1], lab
    p_lab = pl.copy()
    p_lab[p_lab > 0.5] = 1
    p_lab[p_lab <= 0.5] = 0
    dif = 2 * gr - p_lab

    # calculate
    fn, tp, tn, fp = len(np.where(dif == 2)[0]), len(np.where(dif == 1)[0]), len(np.where(dif == 0)[0]), len(np.where(dif == -1)[0])
    acc = (tp + tn) / (fn + tp + tn + fp)
    sen = tp / (tp + fn + 1e-5)
    spe = tn / (tn + fp + 1e-5)

    return np.array([Auc, acc, sen, spe])