import numpy as np

def farthest_point_sampling_distmat_boundary(D, k, boundary_points=None, random_init=True, verbose=False):
    """
    Samples points using farthest point sampling using a complete distance matrix
    Parameters
    -------------------------
    D               : (n,n) distance matrix between points
    k               : int - number of points to sample
    boundary_points : list of boundary points => when specified fps points
                      will be as far away as possible from them
    random_init     : Whether to sample the first point randomly or to
                      take the furthest away from all the other ones
    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """
    if not boundary_points is None:
        dists = D[boundary_points[0]]
        for bpoint in boundary_points:
            dists = np.minimum(dists, D[bpoint])
        inds = boundary_points.tolist()
        dists = 2 * dists
    else:
        if random_init:
            rng = np.random.default_rng()
            inds = [rng.integers(D.shape[0]).item()]
        else:
            inds = [np.argmax(D.sum(1))]

        dists = D[inds[0]]

    iterable = range(k) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, D[newid])
    if boundary_points is not None:
        return np.asarray(inds)[len(boundary_points):]
    else:
        return np.asarray(inds)