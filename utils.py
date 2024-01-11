import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools


def l2_diam(vecs):
    return max([np.linalg.norm(v1-v2, ord=2) for v1, v2 in itertools.combinations(vecs, 2)])


def l2_diam_with_argmax(vecs):
    max_dist, max_pair = -1, None
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            d = np.linalg.norm(vecs[i]-vecs[j], ord=2)
            if d > max_dist:
                max_dist = d
                max_pair = (i, j)
    return max_dist, max_pair


def min_diam_subset(in_vecs, subset_size):
    min_diam = -1
    min_subset = []
    for subset in itertools.combinations(in_vecs, subset_size):
        diam = l2_diam(subset)
        if min_diam == -1 or diam < min_diam:
            min_diam = diam
            min_subset = subset
    return min_diam, min_subset


# True if adding byz_vec to honest_vecs does not increase diameter, False otherwise
def within_diam(honest_vecs, byz_vec, honest_diam):
    return max([np.linalg.norm(h-byz_vec, ord=2) for h in honest_vecs]) <= honest_diam


def trimmed_mean(xs, f):
    return np.mean(sorted(xs) if f == 0 else sorted(xs)[f:-f], axis=0)


def random_orthogonal_to(v, mean=0, std=1):
        r = np.random.normal(mean, std, *v.shape)
        r -= r.dot(v) * v
        return r / np.linalg.norm(r)


def partial_drop(v, p=.5):
    m = np.random.rand(*v.shape) > 1 - p
    return np.ma.array(v, mask=m).filled(fill_value=0)


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T
    dist = xx + yy
    dist.addmm_(1, -2, x, y.T)
    dist[dist < 0] = 0
    return dist.sqrt()