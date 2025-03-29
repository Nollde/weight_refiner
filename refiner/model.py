from collections import defaultdict
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from refiner.data import prepare_data_natural
from refiner.plotting import safe_divide


def prepare_data_reweighter(pos, neg, pos_weights, neg_weights):
    tot = np.concatenate([pos, neg])
    tot_weights = np.concatenate([pos_weights, neg_weights])

    x = np.concatenate([tot, tot])
    y = np.concatenate([np.ones_like(tot_weights), np.zeros_like(tot_weights)])
    w = np.concatenate([tot_weights, np.ones_like(tot_weights)])
    return x, y, w


def prepare_data_refiner(pos, neg, pos_weights, neg_weights):
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos_weights), np.zeros_like(neg_weights)])
    w = np.concatenate([pos_weights, -neg_weights])
    return x, y, w


def split(
    *arrays,
    test_size=0.2,
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


def simple_model(input_shape=(1,), n_layers=2, n_nodes=128):
    """
    Create a simple feedforward neural network model.
    input_shape: shape of the input data (default is (1,))
    n_layers: number of hidden layers (default is 2)
    n_nodes: number of nodes in each hidden layer (default is 128)
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if n_nodes < 1:
        raise ValueError("n_nodes must be at least 1")
    layers = [tf.keras.Input(shape=input_shape)]
    for _ in range(n_layers):
        layers.append(tf.keras.layers.Dense(n_nodes, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    return tf.keras.Sequential(layers=layers)


def apply_reweighter(*data, reweighter=None):
    x, y, _ = prepare_data_natural(*data)
    pred = reweighter.predict(x, batch_size=10_000)[:, 0]
    w_new = pred / (1.0 - pred)
    return x, y, w_new


def apply_refiner(*data, refiner=None):
    x, y, w = prepare_data_refiner(*data)
    pred = refiner.predict(x, batch_size=10_000)[:, 0]
    ratio = 1 / pred - 1
    w_new = w * (1 - ratio) / (1 + ratio)
    return x, y, w_new


def resample(*data):
    """
    Makes simplifying assumption that all weights are +-1
    to calculate variance (see https://arxiv.org/abs/2007.11586)
    """
    x, y, w = data
    keep_probability = w**2
    if not np.all(keep_probability <= 1.0):
        print("WARNING: Probabilities should be <= 1.0")
        keep_probability[keep_probability > 1.0] = 1.0
    keep = np.random.binomial(1, keep_probability) == 1

    x = x[keep]
    y = y[keep]
    w = safe_divide(1.0, w[keep])

    return x, y, w


class HistoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_every_n_batches=None, val_data=None):
        self.log_every_n_batches = log_every_n_batches
        self.val_data = val_data
        self.history = defaultdict(list)
        self.batch_counter = 0

    def perform_validation(self):
        validation = {}
        if self.val_data is not None:
            x_val, y_val, w_val = self.val_data
            val_loss, val_accuracy = self.model.evaluate(
                x_val,
                y_val,
                sample_weight=w_val,
                verbose=0,
            )
            validation["val_loss"] = val_loss
            validation["val_accuracy"] = val_accuracy
        return validation

    def log_to_history(self, logs):
        for key, value in logs.items():
            self.history[key].append((self.batch_counter, value))
        validation = self.perform_validation()
        for key, value in validation.items():
            self.history[key].append((self.batch_counter, value))

    def on_epoch_end(self, epoch, logs=None):
        if self.log_every_n_batches is None:
            self.log_to_history(logs)

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.log_every_n_batches is not None:
            if self.batch_counter % self.log_every_n_batches == 0:
                self.log_to_history(logs)


class InfoLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        optimizer = self.model.optimizer
        if isinstance(optimizer, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = optimizer.learning_rate(optimizer.iterations)
        else:
            lr = optimizer.learning_rate
        print(f"Epoch {epoch+1}, Learning Rate: {lr}")


class SimpleModel:
    def __init__(self, *args, **kwargs):
        self.model = simple_model(*args, **kwargs)

    def compile(
        self,
        *args,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam,
        metrics=["accuracy"],
        learning_rate=None,
        n_train=None,
        epochs=None,
        batch_size=None,
        **kwargs,
    ):
        if isinstance(learning_rate, tuple):
            initial_lr, final_lr = learning_rate
            steps_per_epoch = n_train // batch_size
            learning_rate = self.lr_schedule_exp(
                initial_lr=initial_lr,
                final_lr=final_lr,
                total_epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )
        if learning_rate is None:
            optimizer = optimizer()
        else:
            optimizer = optimizer(learning_rate=learning_rate)
        self.model.compile(
            *args,
            **kwargs,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        return self

    def fit(self, *args, **kwargs):
        self.logger = HistoryLogger(
            val_data=kwargs.get("validation_data"),
        )
        self.model.fit(
            *args,
            **kwargs,
            callbacks=[
                self.logger,
                # InfoLogger(),
            ],
        )
        return self.logger

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def reduce_lr_plateau(self, factor=0.2, patience=50):
        return tf.keras.callbacks.ReduceLROnPlateau(
            factor=factor,
            patience=patience,
        )

    def lr_schedule_exp(self, initial_lr, final_lr, total_epochs, steps_per_epoch):
        total_decay = final_lr / initial_lr
        decay_rate = total_decay ** (1 / total_epochs)

        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=steps_per_epoch,
            decay_rate=decay_rate,
            staircase=True,
        )

    def early_stopping(self, patience=1000):
        return tf.keras.callbacks.EarlyStopping(
            patience=patience,
        )
