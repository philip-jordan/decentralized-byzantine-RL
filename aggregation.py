from utils import *
import numpy as np


def avg_aggregate(grads):
    return np.mean(np.stack(grads), axis=0).astype(np.float32)


# from https://github.com/epfml/byzantine-robust-noniid-optimizer
def cm_aggregate(grads):
    return np.median(np.stack(grads), axis=0).astype(np.float32)


# from https://github.com/epfml/byzantine-robust-noniid-optimizer
def rfa_aggregate(grads, T=8, nu=0.1):
    alphas = [1 / len(grads) for _ in grads]
    z = np.zeros_like(grads[0])

    m = len(grads)
    if len(alphas) != m:
        raise ValueError

    if nu < 0:
        raise ValueError

    for _ in range(T):
        betas = []
        for k in range(m):
            distance = np.linalg.norm(z - grads[k], 2)
            betas.append(alphas[k] / max(distance, nu))

        z = np.zeros(grads[0].shape, dtype=np.float32)
        for w, beta in zip(grads, betas):
            z += w * beta
        z /= sum(betas)
    return z


# from https://github.com/epfml/byzantine-robust-noniid-optimizer
def krum_aggregate(grads, opts):
    def pairwise_euclidean_distances(vectors):
        """Compute the pairwise euclidean distance.
        Arguments:
            vectors {list} -- A list of vectors.
        Returns:
            dict -- A dict of dict of distances {i:{j:distance}}
        """
        n = len(vectors)
        distances = {}
        for i in range(n - 1):
            distances[i] = {}
            for j in range(i + 1, n):
                distances[i][j] = np.linalg.norm(vectors[i] - vectors[j], 2) ** 2
        return distances

    def _compute_scores(distances, i, n, f):
        """Compute scores for node i.
        Arguments:
            distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
            i {int} -- index of worker, starting from 0.
            n {int} -- total number of workers
            f {int} -- Total number of Byzantine workers.
        Returns:
            float -- krum distance score of i.
        """
        s = [distances[j][i] ** 2 for j in range(i)] + [
            distances[i][j] ** 2 for j in range(i + 1, n)
        ]
        _s = sorted(s)[: n - f - 2]
        return sum(_s)

    def multi_krum(distances, n, f, m):
        """Multi_Krum algorithm
        Arguments:
            distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
            n {int} -- Total number of workers.
            f {int} -- Total number of Byzantine workers.
            m {int} -- Number of workers for aggregation.
        Returns:
            list -- A list indices of worker indices for aggregation. length <= m
        """
        if n < 1:
            raise ValueError(
                "Number of workers should be positive integer. Got {}.".format(f)
            )

        if m < 1 or m > n:
            raise ValueError(
                "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                    m, n
                )
            )

        if 2 * f + 2 > n:
            raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

        for i in range(n - 1):
            for j in range(i + 1, n):
                if distances[i][j] < 0:
                    raise ValueError(
                        "The distance between node {} and {} should be non-negative: Got {}.".format(
                            i, j, distances[i][j]
                        )
                    )

        scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
        sorted_scores = sorted(scores, key=lambda x: x[1])
        return list(map(lambda x: x[0], sorted_scores))[:m]

    distances = pairwise_euclidean_distances(grads)
    top_m_indices = multi_krum(distances, opts.num_workers, opts.num_byz, opts.num_workers - opts.num_byz - 1)
    return sum(grads[i] for i in top_m_indices) / len(top_m_indices)