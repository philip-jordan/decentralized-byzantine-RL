import numpy as np
from utils import *
 

def simple_average_agree(in_vecs, f):
    return np.mean(in_vecs, axis=0)
 

def heuristic_mda_average_agree(in_vecs, f):
    if len(in_vecs) == 1:
        return in_vecs[0]
    vecs = list(in_vecs)
    init_len = len(vecs)
    while len(vecs) > init_len - f:
        diam, (i, j) = l2_diam_with_argmax(vecs)
        i, j = min(i, j), max(i, j)
        v1 = vecs.pop(j)
        d1 = l2_diam(vecs)
        vecs.append(v1)
        v2 = vecs.pop(i)
        d2 = l2_diam(vecs)
        if d1 < d2:
            vecs[-1] = v2
    return np.mean(vecs, axis=0)


def mda_average_agree(in_vecs, f, byz_dir=None, opts=None):
    if len(in_vecs) == 1:
        return in_vecs[0]

    if byz_dir is not None or opts.attack_type == 'mda_fooling_orth':
        diam_honest, min_subset = min_diam_subset(in_vecs[f:], len(in_vecs[f:]) - f)
        honest_mean = np.mean(min_subset, axis=0)
        byz_vec = random_orthogonal_to(honest_mean) if opts.attack_type == 'mda_fooling_orth' else byz_dir
        l, r = 0, 4*diam_honest
        while abs(l-r) > 0.00001:
            m = (l+r)/2
            if within_diam(min_subset, honest_mean + m*byz_vec, diam_honest):
                l = m
            else:
                r = m
        new_vecs = [(in_vecs[i] if i >= f else (honest_mean + m*byz_vec)) for i in range(len(in_vecs))]
    else:
        new_vecs = [in_vecs[i] for i in range(len(in_vecs))]

    _, min_subset = min_diam_subset(new_vecs, len(new_vecs) - f)
    return np.mean(min_subset, axis=0)


def rbtm_average_agree(in_vecs, f):
    if len(in_vecs) == 1:
        return in_vecs[0]
    res = np.zeros(in_vecs[0].shape)
    for i in range(len(res)):
        res[i] = trimmed_mean([v[i] for v in in_vecs], f)
    return res.astype(np.float32)