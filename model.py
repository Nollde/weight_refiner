import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data import prepare_data_natural

def prepare_data_resampler(pos, neg, pos_weights, neg_weights):
    tot = np.concatenate([pos, neg])
    tot_weights = np.concatenate([pos_weights, neg_weights])

    x = np.concatenate([tot, tot])
    y = np.concatenate([np.ones_like(tot_weights), np.zeros_like(tot_weights)])
    w = np.concatenate([tot_weights, np.ones_like(tot_weights)])
    return x, y, w


def prepare_data_reweighter(pos, neg, pos_weights, neg_weights):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos_weights), np.zeros_like(neg_weights)])
    w = np.concatenate([pos_weights, -neg_weights])
    return x, y, w


def split(
    *arrays,
    test_size=0.5,
    random_state=42,
    **kwargs,
):
    return train_test_split(
        *arrays,
        test_size=test_size,
        random_state=random_state,
        **kwargs,
    )


def get_train(*data, **kwargs):
    x_train, _, y_train, __, w_train, ___ = split(*data, **kwargs)
    return x_train, y_train, w_train


def get_val(*data, **kwargs):
    _, x_val, __, y_val, ___, w_val = split(*data, **kwargs)
    return x_val, y_val, w_val


def simple_model(input_shape=(1,)):
    return tf.keras.Sequential(
        layers=[
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def apply_reweighter(*data, reweighter):
    x, y, w = prepare_data_reweighter(*data)
    pred = reweighter.predict(x, batch_size=10_000)[:, 0]
    ratio = 1 / pred - 1
    w_new = w * (1 - ratio) / (1 + ratio)
    return x, y, w_new


# def check_reweighter(vals, reweighter):
#     # Analytic check of reweighter ratio
#     # vals = (bins[1:] + bins[:-1]) / 2
#     ratio = 1 / reweighter.predict(vals) - 1

#     plt.plot(vals, ratio, label="DNN Ratio")
#     plt.plot(vals, np.abs(neg)/pos, label="Analytic Ratio")
#     plt.legend()


def apply_resampler(*data, resampler):
    x, y, _ = prepare_data_natural(*data)
    pred = resampler.predict(x, batch_size=10_000)[:, 0]
    w_new = pred / (1.0 - pred)
    return x, y, w_new


def resample(*data):
    x, y, w = data
    keep_probability = w ** 2
    assert np.all(keep_probability <= 1.0), "Probabilities must be <= 1.0"
    keep = np.random.binomial(1, keep_probability) == 1

    x = x[keep]
    y = y[keep]
    w = 1 / (0.00001 + w[keep])

    return x, y, w