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
    shape=(1,),
):
    """
    n_total: total number of samples
    neg_frac: fraction of (unweighted) negative samples
    neg_weight: weight of negative samples
    """

    n_neg = int(n_total * neg_frac)
    n_pos = n_total - n_neg

    pos = np.random.normal(loc=pos_loc, scale=pos_scale, size=(n_pos,) + shape)
    neg = np.random.normal(loc=neg_loc, scale=neg_scale, size=(n_neg,) + shape)

    pos_weights = np.random.normal(loc=1, scale=weight_scale, size=n_pos)
    neg_weights = np.random.normal(loc=neg_weight, scale=weight_scale, size=n_neg)

    return pos, neg, pos_weights, neg_weights


def create_data_function(n_pos, n_neg=0, shape=(1,), function_pos=lambda x: x, function_neg=lambda x: x):
    """
    Create data based on a function for positive and negative samples.
    n_pos: number of positive samples
    n_neg: number of negative samples (default is 0)
    shape: shape of each sample (default is (1,))
    function_pos: function to generate positive weights (default is identity)
    function_neg: function to generate negative weights (default is identity)
    Note: The first dimension of the shape is the sample size, and the rest are the features.
    """

    pos = np.random.uniform(0, 3, size=(n_pos,) + shape)
    pos_weights = function_pos(pos[:, 0])
    neg = np.random.uniform(0, 3, size=(n_neg,) + shape)
    neg_weights = function_neg(neg[:, 0])

    return pos, neg, pos_weights, neg_weights


def load_data_tt(n_jets=1):
    input_dir = "/Users/dnoll/projects/NeuralPositiveResampler"  # Local path
    x = np.load(f"{input_dir}/x.npy")
    w = np.load(f"{input_dir}/w.npy")

    # get only first jet
    x = x[:, :n_jets].reshape(x.shape[0], -1)

    pos_mask = w >= 0
    neg_mask = w < 0

    pos = x[pos_mask]
    neg = x[neg_mask]

    pos_weights = w[pos_mask]
    neg_weights = w[neg_mask]

    return pos, neg, pos_weights, neg_weights


def prepare_data_natural(pos, neg, pos_weights, neg_weights):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos_weights), np.zeros_like(neg_weights)])
    w = np.concatenate([pos_weights, neg_weights])
    return x, y, w
