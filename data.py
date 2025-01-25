import numpy as np

def create_data_gaussian(
    n_total,
    neg_frac=0.1,
    neg_weight=-1,
    weight_scale=0.0,
    pos_loc=0,
    pos_scale=1,
    neg_loc=0,
    neg_scale=0.5,
):
    """
    n_total: total number of samples
    neg_frac: fraction of (unweighted) negative samples
    neg_weight: weight of negative samples
    """

    n_neg = int(n_total * neg_frac)
    n_pos = n_total - n_neg

    pos = np.random.normal(loc=pos_loc, scale=pos_scale, size=n_pos)
    neg = np.random.normal(loc=neg_loc, scale=neg_scale, size=n_neg)
    
    pos_weights = np.random.normal(loc=1, scale=weight_scale, size=n_pos)
    neg_weights = np.random.normal(loc=neg_weight, scale=weight_scale, size=n_neg)

    return pos, neg, pos_weights, neg_weights



    return pos, neg, pos_weights, neg_weights


def prepare_data_natural(pos, neg, pos_weights, neg_weights):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos_weights), np.zeros_like(neg_weights)])
    w = np.concatenate([pos_weights, neg_weights])
    return x, y, w