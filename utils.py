import math
import numpy as np
import cv2
import random
import pyemd
import scipy.misc

def nss(pred_sal, fix_map):
    fix_map = fix_map.astype(np.bool)
    pred_sal = (pred_sal - np.mean(pred_sal)) / np.std(pred_sal)
    return np.mean(pred_sal[fix_map])


def auc_judd(pred_sal, fix_map, jitter=True):
    fix_map = fix_map.flatten().astype(np.bool)
    pred_sal = pred_sal.flatten().astype(np.float)
    if jitter:
        jitter = np.random.rand(pred_sal.shape[0]) / 1e7
        pred_sal += jitter
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    all_thres = np.sort(pred_sal[fix_map])[::-1]

    tp = np.concatenate([[0], np.linspace(0.0, 1.0, all_thres.shape[0]), [1]])
    fp = np.zeros((all_thres.shape[0]))
    sorted_sal = np.sort(pred_sal)
    for ind, thres in enumerate(all_thres):
        above_thres = sorted_sal.shape[0] - sorted_sal.searchsorted(thres, side='left')
        fp[ind] = (above_thres - ind) * 1. / (pred_sal.shape[0] - all_thres.shape[0])
    fp = np.concatenate([[0], fp, [1]])
    return np.trapz(tp, fp)


def auc_borji(pred_sal, fix_map, n_split=100, step_size=.1):
    fix_map = fix_map.flatten().astype(np.bool)
    pred_sal = pred_sal.flatten().astype(np.float)
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    sal_fix = pred_sal[fix_map]
    sorted_sal_fix = np.sort(sal_fix)

    r = np.random.randint(0, pred_sal.shape[0], (sal_fix.shape[0], n_split))
    rand_fix = pred_sal[r]
    auc = np.zeros((n_split))
    for i in range(n_split):
        cur_fix = rand_fix[:, i]
        sorted_cur_fix = np.sort(cur_fix)
        max_val = np.maximum(cur_fix.max(), sal_fix.max())
        tmp_all_thres = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros((tmp_all_thres.shape[0]))
        fp = np.zeros((tmp_all_thres.shape[0]))
        for ind, thres in enumerate(tmp_all_thres):
            tp[ind] = (sorted_sal_fix.shape[0] - sorted_sal_fix.searchsorted(thres, side='left')) * 1. / sal_fix.shape[
                0]
            fp[ind] = (sorted_cur_fix.shape[0] - sorted_cur_fix.searchsorted(thres, side='left')) * 1. / sal_fix.shape[
                0]
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)


def cc(pred_sal, gt_sal):
    pred_sal = (pred_sal - pred_sal.mean()) / (pred_sal.std())
    gt_sal = (gt_sal - gt_sal.mean()) / (gt_sal.std())
    return np.corrcoef(pred_sal.flat, gt_sal.flat)[0, 1]


def sim(pred_sal, gt_sal):
    pred_sal = pred_sal.astype(np.float)
    gt_sal = gt_sal.astype(np.float)
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    pred_sal = pred_sal / pred_sal.sum()
    gt_sal = (gt_sal - gt_sal.min()) / (gt_sal.max() - gt_sal.min())
    gt_sal = gt_sal / gt_sal.sum()
    diff = np.minimum(pred_sal, gt_sal)
    return np.sum(diff)


def kl(pred_sal, fix_map):
    eps = np.finfo(float).eps
    pred_sal = pred_sal.astype(np.float)
    fix_map = fix_map.astype(np.float)
    pred_sal = pred_sal / pred_sal.sum()
    fix_map = fix_map / fix_map.sum()
    return np.sum(fix_map * np.log(eps + fix_map / (pred_sal + eps)))


def ig(pred_sal, fix_map, base_sal):
    eps = np.finfo(float).eps
    fix_map = fix_map.astype(np.bool)
    pred_sal = pred_sal.astype(np.float32).flatten()
    base_sal = base_sal.astype(np.float32).flatten()
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    base_sal = (base_sal - base_sal.min()) / (base_sal.max() - base_sal.min())
    pred_sal = pred_sal / pred_sal.sum()
    base_sal = base_sal / base_sal.sum()
    locs = fix_map.flatten()
    return np.mean(np.log2(eps + pred_sal[locs]) - np.log2(eps + base_sal[locs]))


def auc_shuffled(pred_sal, fix_map, base_map, n_split=10, step_size=.1):

    assert (base_map.shape == fix_map.shape)
    pred_sal = pred_sal.flatten().astype(np.float)
    base_map = base_map.flatten().astype(np.float)
    fix_map = fix_map.flatten().astype(np.bool)
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    sal_fix = pred_sal[fix_map]
    sorted_sal_fix = np.sort(sal_fix)
    ind = np.where(base_map > 0)[0]
    n_fix = sal_fix.shape[0]
    n_fix_oth = np.minimum(n_fix, ind.shape[0])

    rand_fix = np.zeros((n_fix_oth, n_split))
    for i in range(n_split):
        rand_ind = random.sample(list(ind), n_fix_oth)
        rand_fix[:, i] = pred_sal[rand_ind]
    auc = np.zeros((n_split))
    for i in range(n_split):
        cur_fix = rand_fix[:, i]
        sorted_cur_fix = np.sort(cur_fix)
        max_val = np.maximum(cur_fix.max(), sal_fix.max())
        tmp_all_thres = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros((tmp_all_thres.shape[0]))
        fp = np.zeros((tmp_all_thres.shape[0]))
        for ind, thres in enumerate(tmp_all_thres):
            tp[ind] = (sorted_sal_fix.shape[0] - sorted_sal_fix.searchsorted(thres, side='left')) * 1. / n_fix
            fp[ind] = (sorted_cur_fix.shape[0] - sorted_cur_fix.searchsorted(thres, side='left')) * 1. / n_fix_oth
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)


def emd(pred_sal, fix_map, downsize=32):
    pred_sal = cv2.resize(pred_sal, None,fx=1/downsize, fy=1 / downsize)
    fix_map = cv2.resize(fix_map, None, fx=1 / downsize, fy=1 / downsize)
    fix_map = (fix_map - fix_map.min()) / (fix_map.max() - fix_map.min())
    pred_sal = pred_sal.astype(np.float)
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())

    fix_map = fix_map / fix_map.sum()
    pred_sal = pred_sal / pred_sal.sum()

    c, r = fix_map.shape
    N = r * c
    dist = np.zeros((N, N), dtype=np.float)

    j = 0
    for c1 in range(c):
        for r1 in range(r):
            j = j + 1
            i = 0
            for c2 in range(c):
                for r2 in range(r):
                    i = i + 1
                    dist[i - 1, j - 1] = math.sqrt((r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2))
    p = pred_sal.flatten()
    q = fix_map.flatten()
    return pyemd.emd(p, q, dist, extra_mass_penalty=0.)

