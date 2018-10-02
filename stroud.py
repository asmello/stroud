import numpy as np
import tensorflow as tf

import warnings
from collections import defaultdict

class SimpleAgent:

    def __init__(self, layer_width=16, n_layers=1, dropout=0, loss='mse'):
        self.layer_width = layer_width
        self.n_layers = n_layers
        self.dropout = dropout
        self.loss_type = loss
        self._ready = False

    def _build(self):
        tf.reset_default_graph()
        self._x_in = tf.placeholder(tf.float32, shape=(None, self.n_features), name='features_in')
        self._p_in = tf.placeholder(tf.float32, shape=(None,), name='parameter_in')
        self._r_in = tf.placeholder_with_default(tf.zeros_like(self._p_in),
                                                 shape=(None,), name='rewards_in')
        self._train = tf.placeholder(tf.bool, shape=(), name='train_mode_switch')
        self._batch_size = tf.placeholder_with_default(tf.zeros((), dtype=tf.int64),
                                                       shape=(), name='batch_size')
        self._n_epochs = tf.placeholder_with_default(tf.zeros((), dtype=tf.int64),
                                                     shape=(), name='n_epochs')
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        x_cat = tf.concat([self._x_in, tf.reshape(self._p_in, (-1, 1))],
                          axis=-1, name='merge_features')

        def train_path():
            it = (tf.data.Dataset
                         .from_tensor_slices((x_cat, self._r_in))
                         .shuffle(buffer_size=100000)
                         .batch(self._batch_size)
                         .repeat(self._n_epochs)
                    .make_initializable_iterator())
            return (*it.get_next(), it.initializer)

        def pred_path():
            it = (tf.data.Dataset
                         .from_tensor_slices((x_cat, self._r_in))
                         .batch(self._batch_size)
                    .make_initializable_iterator())
            return (*it.get_next(), it.initializer)

        x, r, self._it_init_op = tf.cond(self._train, train_path, pred_path)

        for i in range(self.n_layers):
            x = tf.layers.Dense(self.layer_width, activation='elu', name=f'layer_{i}')(x)
            if self.dropout > 0:
                x = tf.layers.Dropout(self.dropout)(x)

        if self.loss_type == 'mse':
            self._r_out = tf.squeeze(tf.layers.Dense(1, name=f'rewards_out')(x))
            self._loss = tf.losses.mean_squared_error(r, self._r_out)
        elif self.loss_type == 'ce':
            logits = tf.squeeze(tf.layers.Dense(1, name=f'rewards_out')(x))
            self._r_out = tf.sigmoid(logits)
            self._loss = tf.losses.sigmoid_cross_entropy(r, logits)
        else:
            raise RuntimeError("Invalid loss type.")

        self._train_op = (tf.train.AdamOptimizer()
                                  .minimize(self._loss, global_step=self._global_step))

    def fit(self, X, p, r, bins=None, batch_size=32, n_epochs=1):

        X = np.asanyarray(X, dtype=np.float32)
        p = np.asanyarray(p, dtype=np.float32)
        r = np.asanyarray(r, dtype=np.float32)

        assert X.ndim == 2, "Incompatible features tensor, expected rank 2 (i.e. a matrix)."
        assert p.ndim == 1, "Incompatible parameters tensor, expected rank 1 (i.e. a vector)."
        assert r.ndim == 1, "Incompatible rewards tensor, expected rank 1 (i.e. a vector)."

        xlen, n_features = X.shape

        if not self._ready or n_features != self.n_features:
            if self._ready:
                warnings.warn("Replacing model due to new X shape.", RuntimeWarning)

            self.n_features = n_features
            self._build()

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            self._ready = True

        self._sess.run(self._it_init_op, feed_dict={
            self._x_in: X, self._p_in: p, self._r_in: r,
            self._train: True, self._batch_size: batch_size, self._n_epochs: n_epochs
        })

        batches_per_epoch = np.ceil(xlen / batch_size).astype(int)

        losses = []
        while True:
            try:
                step, loss, _ = self._sess.run(
                    [self._global_step, self._loss, self._train_op],
                    feed_dict={self._train: True}
                )
                losses.append(loss)
                if step % batches_per_epoch == 0:
                    epoch = step // batches_per_epoch
                    print(f"@{epoch}: loss={np.mean(losses)}")
                    losses = []
            except tf.errors.OutOfRangeError:
                break

    def predict(self, X, p, batch_size=128):

        assert self._ready, "Must fit model first."

        self._sess.run(self._it_init_op, feed_dict={
            self._x_in: X,
            self._p_in: p,
            self._train: False,
            self._batch_size: batch_size
        })

        batches = []
        while True:
            try:
                batches.append(self._sess.run(self._r_out, feed_dict={self._train: False}))
            except tf.errors.OutOfRangeError:
                break

        return np.concatenate(batches, axis=0)

    def estimate_curve(self, X, span, n_samples=100, batch_size=128):
        space = np.linspace(*span, n_samples)
        X_long = np.tile(X, n_samples).reshape(-1, X.shape[1])
        p_long = np.tile(space, (X.shape[0], 1)).flatten()
        return space, self.predict(X_long, p_long, batch_size).reshape(-1, n_samples)

    def estimate_best_p(self, X, span, n_samples=100, batch_size=128):
        space, y_curve = self.estimate_curve(X, span, n_samples, batch_size)
        return space[np.argmax(y_curve, axis=1)]

    def evaluate(self, X, p, y, batch_size=128, eps=1e-5):
        """Evaluate the correctness of the estimated rewards.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p : ndarray
            The parameters tried (may be more than 1 for each X row).
        y : ndarray
            The true rewards (same shape as p).
        batch_size : int
            The batch size for the prediction pipeline.
        eps : float, optional
            Small positive number to compute cross entropy with.
            Only meaningful when loss type is 'ce'.

        Returns
        -------
        float
            The loss over the estimated and true rewards.
        """
        if self.loss_type == 'ce':
            y_pred = np.clip(self.predict(X, p, batch_size), eps, 1-eps)
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        elif self.loss_type == 'mse':
            return np.mean((y - self.predict(X, p, batch_size)) ** 2)

class FiLMLayer(tf.keras.layers.Layer):

    def __init__(self, p, axis=-1, **kwargs):
        super(FiLMLayer, self).__init__(**kwargs)
        self.axis = axis
        self.p = tf.reshape(p, (-1, 1))

    def build(self, input_shape):
        shape = (input_shape[self.axis].value,)
        self.gamma = self.add_variable("gamma", shape=shape, dtype=tf.float32)
        self.beta = self.add_variable("beta", shape=shape, dtype=tf.float32)
        super(FiLMLayer, self).build(input_shape)

    def call(self, inputs):
        return (self.gamma * inputs + self.beta) * self.p

class FiLMAgent:

    def __init__(self, layer_width=16, n_layers=1, dropout=0, loss='mse'):
        self.layer_width = layer_width
        self.n_layers = n_layers
        self.dropout = dropout
        self.loss_type = loss
        self._ready = False

    def _build(self):
        tf.reset_default_graph()
        self._x_in = tf.placeholder(tf.float32, shape=(None, self.n_features), name='features_in')
        self._p_in = tf.placeholder(tf.float32, shape=(None,), name='parameter_in')
        self._r_in = tf.placeholder_with_default(tf.zeros_like(self._p_in),
                                                 shape=(None,), name='rewards_in')
        self._train = tf.placeholder(tf.bool, shape=(), name='train_mode_switch')
        self._batch_size = tf.placeholder_with_default(tf.zeros((), dtype=tf.int64),
                                                       shape=(), name='batch_size')
        self._n_epochs = tf.placeholder_with_default(tf.zeros((), dtype=tf.int64),
                                                     shape=(), name='n_epochs')
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        def train_path():
            it = (tf.data.Dataset
                         .from_tensor_slices((self._x_in, self._p_in, self._r_in))
                         .shuffle(buffer_size=100000)
                         .batch(self._batch_size)
                         .repeat(self._n_epochs)
                    .make_initializable_iterator())
            return (*it.get_next(), it.initializer)

        def pred_path():
            it = (tf.data.Dataset
                         .from_tensor_slices((self._x_in, self._p_in, self._r_in))
                         .batch(self._batch_size)
                    .make_initializable_iterator())
            return (*it.get_next(), it.initializer)

        x, p, r, self._it_init_op = tf.cond(self._train, train_path, pred_path)
        film_layer = FiLMLayer(p)

        for i in range(self.n_layers):
            x = tf.layers.Dense(self.layer_width, activation='elu', name=f'layer_{i}')(x)
            if self.dropout > 0:
                x = tf.layers.Dropout(self.dropout)(x)
            x = film_layer(x)

        if self.loss_type == 'mse':
            self._r_out = tf.squeeze(tf.layers.Dense(1, name=f'rewards_out')(x))
            self._loss = tf.losses.mean_squared_error(r, self._r_out)
        elif self.loss_type == 'ce':
            logits = tf.squeeze(tf.layers.Dense(1, name=f'rewards_out')(x))
            self._r_out = tf.sigmoid(logits)
            self._loss = tf.losses.sigmoid_cross_entropy(r, logits)
        else:
            raise RuntimeError("Invalid loss type.")

        self._train_op = (tf.train.AdamOptimizer()
                                  .minimize(self._loss, global_step=self._global_step))

    def fit(self, X, p, r, bins=None, batch_size=32, n_epochs=1):

        X = np.asanyarray(X, dtype=np.float32)
        p = np.asanyarray(p, dtype=np.float32)
        r = np.asanyarray(r, dtype=np.float32)

        assert X.ndim == 2, "Incompatible features tensor, expected rank 2 (i.e. a matrix)."
        assert p.ndim == 1, "Incompatible parameters tensor, expected rank 1 (i.e. a vector)."
        assert r.ndim == 1, "Incompatible rewards tensor, expected rank 1 (i.e. a vector)."

        xlen, n_features = X.shape

        if not self._ready or n_features != self.n_features:
            if self._ready:
                warnings.warn("Replacing model due to new X shape.", RuntimeWarning)

            self.n_features = n_features
            self._build()

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            self._ready = True

        self._sess.run(self._it_init_op, feed_dict={
            self._x_in: X, self._p_in: p, self._r_in: r,
            self._train: True, self._batch_size: batch_size, self._n_epochs: n_epochs
        })

        batches_per_epoch = np.ceil(xlen / batch_size).astype(int)

        losses = []
        while True:
            try:
                step, loss, _ = self._sess.run(
                    [self._global_step, self._loss, self._train_op],
                    feed_dict={self._train: True}
                )
                losses.append(loss)
                if step % batches_per_epoch == 0:
                    epoch = step // batches_per_epoch
                    print(f"@{epoch}: loss={np.mean(losses)}")
                    losses = []
            except tf.errors.OutOfRangeError:
                break

    def predict(self, X, p, batch_size=128):

        assert self._ready, "Must fit model first."

        self._sess.run(self._it_init_op, feed_dict={
            self._x_in: X,
            self._p_in: p,
            self._train: False,
            self._batch_size: batch_size
        })

        batches = []
        while True:
            try:
                batches.append(self._sess.run(self._r_out, feed_dict={self._train: False}))
            except tf.errors.OutOfRangeError:
                break

        return np.concatenate(batches, axis=0)

    def estimate_curve(self, X, span, n_samples=100, batch_size=128):
        space = np.linspace(*span, n_samples)
        X_long = np.tile(X, n_samples).reshape(-1, X.shape[1])
        p_long = np.tile(space, (X.shape[0], 1)).flatten()
        return space, self.predict(X_long, p_long, batch_size).reshape(-1, n_samples)

    def estimate_best_p(self, X, span, n_samples=100, batch_size=128):
        space, y_curve = self.estimate_curve(X, span, n_samples, batch_size)
        return space[np.argmax(y_curve, axis=1)]

    def evaluate(self, X, p, y, batch_size=128, eps=1e-5):
        """Evaluate the correctness of the estimated rewards.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p : ndarray
            The parameters tried (may be more than 1 for each X row).
        y : ndarray
            The true rewards (same shape as p).
        batch_size : int
            The batch size for the prediction pipeline.
        eps : float, optional
            Small positive number to compute cross entropy with.
            Only meaningful when loss type is 'ce'.

        Returns
        -------
        float
            The loss over the estimated and true rewards.
        """
        if self.loss_type == 'ce':
            y_pred = np.clip(self.predict(X, p, batch_size), eps, 1-eps)
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        elif self.loss_type == 'mse':
            return np.mean((y - self.predict(X, p, batch_size)) ** 2)

class MultiHeadAgent:
    """Multiheaded neural network that attempts to maximize reward.

    Attributes
    ----------
    dropout : float
        Probability of inhibiting output from a neuron during training.
    n_heads : int
        Number of heads that estimate reward.
    body_layers : int
        Number of layers in the network's body.
    layer_width : int
        Number of neurons in each layer.
    record_precision : int
        Round features to this number of decimal places when keeping history.
    max_history : int
        How many previous parameters to remember during active learning.
    loss_type : str
        Which kind of loss to use for optimization. Accepted values: ['mse', 'ce'].
    """

    @property
    def dropout(self):
        return self._dropout

    @property
    def n_heads(self):
        return self._n_heads

    @property
    def body_layers(self):
        return self._body_layers

    @property
    def layer_width(self):
        return self._layer_width

    @property
    def record_precision(self):
        return self._record_precision

    @property
    def max_history(self):
        return self._max_history

    @property
    def loss_type(self):
        return self._loss_type

    def __init__(self, layer_width=16, body_layers=1, dropout=0, n_heads=10,
                 max_history=100, record_precision=4, loss='ce'):
        assert loss in ['mse', 'ce'], "Invalid loss type."
        self._record_precision = record_precision
        self._max_history = max_history
        self._layer_width = layer_width
        self._body_layers = body_layers
        self._dropout = dropout
        self._n_heads = n_heads
        self._loss_type = loss
        self._ready = False
        self._H = defaultdict(list)
        self._populate_layer_cache()

    def _populate_layer_cache(self):
        """Generate and cache the layers, so they can be shared along the parameter axis."""
        self._layer_cache = {}
        for i in range(self.body_layers):
            self._layer_cache[f'layer_{i}'] = \
                tf.layers.Dense(self.layer_width, activation='elu', name=f'layer_{i}')
            if self.dropout > 0:
                self._layer_cache[f'layer_{i}_dropout'] = \
                    tf.layers.Dropout(self.dropout, name=f'layer_{i}_dropout')
        for i in range(self.n_heads):
            self._layer_cache[f'head_{i}'] = \
                tf.layers.Dense(self.layer_width, activation='elu', name=f'head_{i}')
            self._layer_cache[f'head_{i}_output'] = \
                tf.layers.Dense(1, name=f'head_{i}_output')

    def _build(self):
        """Build the TensorFlow computation graph."""
        self._x_in = tf.placeholder(tf.float32, shape=(None, self.n_features), name='features')
        self._p_in = tf.placeholder(tf.float32, shape=(None, None), name='parameters')
        self._y_in = tf.placeholder(tf.float32, shape=(None, None), name='true_rewards')
        h_default = tf.random_uniform((), maxval=self.n_heads,
                                      dtype=tf.int32, name='random_head')
        self._h_sel = tf.placeholder_with_default(h_default, shape=(), name='head_selector')

        # Applies the shared model for a single transposed column of p_in
        def apply_model(p):
            z = tf.concat([self._x_in, tf.reshape(p, (-1, 1))], axis=-1)
            for i in range(self.body_layers):
                z = self._layer_cache[f'layer_{i}'](z)
                if self.dropout > 0:
                    z = self._layer_cache[f'layer_{i}_dropout'](z)
            outputs = []
            for i in range(self.n_heads):
                z = self._layer_cache[f'head_{i}'](z)
                outputs.append(self._layer_cache[f'head_{i}_output'](z))
            return outputs

        # apply model for each parameter (p_in columns)
        # -- transposed because map_fn runs along rows
        h_out_list = tf.map_fn(apply_model, tf.transpose(self._p_in),
                               dtype=[tf.float32]*self.n_heads)

        # h_out is (n_samples, n_parameters, n_heads) and gives estimated logit-rewards
        with tf.name_scope("heads_output"):
            h_out = tf.transpose(tf.concat(h_out_list, axis=-1), perm=[1, 0, 2])

            if self.loss_type == 'ce':
                self._h_out = tf.sigmoid(h_out)
            elif self.loss_type == 'mse':
                self._h_out = h_out

            self._h_mean = tf.reduce_mean(self._h_out, axis=-1, name="heads_mean")
            self._mbest_p = tf.argmax(self._h_mean, axis=-1, name="best_parameters")

        # reward estimated by a single head
        with tf.name_scope("selected_output"):
            y_out = h_out[..., self._h_sel]

            if self.loss_type == 'ce':
                self._y_out = tf.sigmoid(y_out)
            elif self.loss_type == 'mse':
                self._y_out = y_out

            self._best_p = tf.argmax(self._y_out, axis=-1, name="best_parameters")

        # TODO: implement sampling heads independently for each row

        with tf.name_scope("training"):
            if self.loss_type == 'ce':
                self._loss = tf.losses.sigmoid_cross_entropy(self._y_in, y_out)
            elif self.loss_type == 'mse':
                self._loss = tf.losses.mean_squared_error(self._y_in, y_out)
            self._train_op = tf.train.AdamOptimizer().minimize(self._loss)

    def bootstrap(self, X, p, y, batch_size=32, n_epochs=1, early_stop_at=0.0, verbose=0):
        """Train the model with pre-sampled parameters and rewards.

        Parameters
        ----------
        X : array_like
            The features matrix.
        p : array_like
            The parameters tried (may be more than 1 for each X row).
        y : array_like
            The rewards (same shape as p).
        batch_size : int, optional
            Mini-batch size used for stochastic gradient descent.
        n_epochs : int, optional
            How many times to iterate through the entire dataset.
        early_stop_at : float, optional
            End training early if epoch loss less becomes than this value.
        verbose : int, optional
            Flag for verbosity control (-1 to disable completely).
        """

        X = np.asanyarray(X, dtype=np.float32)
        p = np.asanyarray(p, dtype=np.float32)
        y = np.asanyarray(y, dtype=np.float32)

        assert X.ndim == 2, "Incompatible features tensor, check X shape."

        if p.ndim < 2:
            p = p.reshape(-1, 1)
            assert len(p) == len(X), "Wrong shape for parameters vector."
        else:
            assert p.ndim == 2, "Wrong shape for parameters vector."

        if y.ndim < 2:
            y = y.reshape(-1, 1)
            assert len(y) == len(X), "Wrong shape for rewards vector."
        else:
            assert y.ndim == 2, "Wrong shape for rewards vector."

        # Keep the data to help active training later
        self._update_hist(X, p.squeeze(), y.squeeze())

        xlen, n_features = X.shape
        bslen = xlen // self.n_heads

        if not self._ready or n_features != self.n_features:
            if self._ready:
                warnings.warn("Replacing model due to new features shape.", RuntimeWarning)

            self.n_features = n_features
            self._build()

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            self._ready = True

        def batch_gen(X_bs, p_bs, y_bs):  # Simple batch generator
            i = 0
            while i + batch_size < bslen:
                span = slice(i, i + batch_size)
                i += batch_size
                yield i, X_bs[span], p_bs[span], y_bs[span]
            if i < bslen:
                yield bslen, X_bs[i:], p_bs[i:], y_bs[i:]

        for epoch in range(n_epochs):

            epoch_losses = []

            # Split training data into bootstraps, one for each head
            for i in range(self.n_heads):

                head_losses = []
                bs = np.random.randint(xlen, size=bslen)

                # Generate batches for this bootstrap, and train with them
                for j, X_b, p_b, y_b in batch_gen(X[bs], p[bs], y[bs]):

                    loss, _ = self._sess.run([self._loss, self._train_op],
                        feed_dict={
                            self._x_in: X_b,
                            self._p_in: p_b,
                            self._y_in: y_b,
                            self._h_sel: i
                        })

                    if verbose > 2:
                        print(f"@{epoch}:H{i}({j}/{len(X_p[i])}) loss={loss}")

                    head_losses.append(loss)

                avg_loss = np.mean(head_losses)

                if verbose > 1:
                    print(f"@{epoch}:H{i} avg_loss={avg_loss}")

                epoch_losses.append(avg_loss)

            avg_loss = np.mean(epoch_losses)

            if verbose > 0:
                print(f"@{epoch}: avg_loss={avg_loss}")

            if avg_loss < early_stop_at:
                if verbose > 0:
                    print("Stopped early (loss target reached).")
                break

        if verbose == 0:
            print(f"Final epoch: avg_loss={avg_loss}")

    def _make_key(self, x):
        return tuple(np.round(x, self.record_precision))

    def _update_hist(self, X, p, y):
        for i in range(len(X)):
            key = self._make_key(X[i])
            hist = self._H[key]
            hist.append((p[i], y[i]))
            self._H[key] = hist[-self.max_history:]  # truncate if too long

    def train(self, X, f, batch_size=32, n_epochs_ep=1, n_episodes=1,
              p_range=(0, 1), sample_density=100, early_stop_at=0.0, verbose=0):
        """Explore the parameter space actively while maximizing cumulative reward.

        Parameters
        ----------
        X : array_like
            The features matrix.
        f : callable
            The reward function.
        batch_size : int, optional
            Mini-batch size used for stochastic gradient descent.
        n_epochs_ep : int, optional
            How many times to iterate through the entire dataset per episode.
        n_episodes : int, optional
            How many feedback loops to run.
        p_range : 2-tuple, optional
            Lower and upper bounds for parameter search.
        sample_density : int, optional
            How many uniformly spaced samples to test at parameter exploration.
        early_stop_at : float, optional
            End episode train-cycle if epoch loss less becomes than this value.
        verbose : int, optional
            Flag for verbosity control (-1 to disable completely).
        """

        X = np.asanyarray(X, dtype=np.float32)

        assert X.ndim == 2, "Incompatible features tensor, check X shape."

        xlen, n_features = X.shape

        if not self._ready or n_features != self.n_features:
            if self._ready:
                warnings.warn("Replacing model due to new features shape.", RuntimeWarning)

            self.n_features = n_features
            self._build()

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            self._ready = True

        # Make sure we have the same amount of parameter
        # samples for every data point
        sample_sizes = [len(self._H[self._make_key(x)]) for x in X]
        if len(set(sample_sizes)) > 1:
            if verbose > 0:
                print("Equalizing samples...")
            n_samples = int(np.median(sample_sizes))
            if n_samples == 0:
                # special case, just drop all samples
                # and start from scratch
                self._H = defaultdict(list)
            else:
                new_samples = 0
                dropped_samples = 0
                for x in X:
                    key = self._make_key(x)
                    hist = self._H[key]
                    size = len(hist)
                    if size > n_samples:
                        # Drop excess samples
                        hist = hist[-n_samples:]
                        dropped_samples += n_samples-size
                    while size < n_samples:
                        # Generate new sample
                        p = np.random.uniform(*p_range, 1)
                        y = f(x.reshape(1, -1), p)
                        hist.append((p.item(), y.item()))
                        new_samples += 1
                        size += 1
                    self._H[key] = hist
                if verbose > 0:
                    print(f"{new_samples} new samples generated.")
                    print(f"{dropped_samples} samples dropped.")

        # NOTE: not using bootstraps because inhomogeneous sampling
        # is bad for training with history
        X_p = np.array_split(np.random.permutation(X), self.n_heads)
        p_p = [self.sample_best_p(X_p[i], p_range, sample_density, use_head=i)
               for i in range(self.n_heads)]

        if verbose > 0:
            print("Fetching initial feedback data...")
        y_p = [f(X_p[i], p_p[i]) for i in range(self.n_heads)]

        for i in range(self.n_heads):  # Update history with feedback
            self._update_hist(X_p[i], p_p[i], y_p[i])

        if verbose > -1:
            mr = np.sum([np.sum(y) for y in y_p]) / xlen
            print(f"> Initial mean reward is {mr:.4f}.")

        # Generate a bootstrap batch using history
        def batch_gen(X_bs):
            i, bslen = 0, len(X_bs)
            while i + batch_size < bslen:
                # Get p, y history for each row in this bootstrap
                hist = np.array([self._H[self._make_key(X_bs[j])]
                                 for j in range(i, i + batch_size)])
                span = slice(i, i + batch_size)
                i += batch_size
                yield i, X_bs[span], hist[..., 0], hist[..., 1]
            if i < bslen:
                hist = np.array([self._H[self._make_key(X_bs[j])]
                                 for j in range(i, bslen)])
                yield bslen, X_bs[i:], hist[..., 0], hist[..., 1]

        for k in range(1, n_episodes+1):

            if verbose > -1:
                print(f"Episode {k} starting...")

            for epoch in range(1, n_epochs_ep+1):

                epoch_losses = []

                for i in range(self.n_heads):

                    head_losses = []

                    for j, X_b, p_b, y_b in batch_gen(X_p[i]):

                        loss, _ = self._sess.run(
                            [self._loss, self._train_op],
                            feed_dict={
                                self._x_in: X_b,
                                self._p_in: p_b,
                                self._y_in: y_b,
                                self._h_sel: i
                            })

                        if verbose > 2:
                            print(f"@{epoch}:H{i}({j}/{len(X_b)}) loss={loss}")

                        head_losses.append(loss)

                    avg_loss = np.mean(head_losses)

                    if verbose > 1:
                        print(f"@{epoch}:H{i} avg_loss={avg_loss}")

                    epoch_losses.append(avg_loss)

                avg_loss = np.mean(epoch_losses)
                std_loss = np.std(epoch_losses)

                if verbose > 0:
                    print(f"@{epoch}: avg_loss={avg_loss:.6f}, std_loss={std_loss:.6f}")

                if avg_loss < early_stop_at:
                    if verbose > 0:
                        print("Stopped early (loss target reached).")
                    break

            if verbose == 0:
                print(f"Final epoch: avg_loss={avg_loss:.6f}, std_loss={std_loss:.6f}")

            X_p = np.array_split(np.random.permutation(X), self.n_heads)

            if verbose > 0:
                print("Re-estimating best parameters...")
            p_p = [self.sample_best_p(X_p[i], p_range, sample_density, use_head=i)
                   for i in range(self.n_heads)]

            if verbose > 0:
                print("Fetching feedback data...")
            y_p = [f(X_p[i], p_p[i]) for i in range(self.n_heads)]

            for i in range(self.n_heads):  # Update history with feedback
                self._update_hist(X_p[i], p_p[i], y_p[i])

            if verbose > -1:
                mr = np.sum([np.sum(y) for y in y_p]) / xlen
                print(f"> After episode {k} mean reward is {mr:.4f}.")

    def sample_best_p(self, X, p_range=(0, 1), sample_density=100, use_head=-1):
        """Sample a range of parameters and return a guess for the best.

        The best guess is sampled from a randomly chosen head.

        Note
        ----
        For now, the same head is sampled for every row in X.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p_range : 2-tuple, optional
            The range used to explore the parameter space.
        sample_density : int, optional
            Number of uniform samples to explore within p_range.
        use_head : int, optional
            Force use of a specific head for sampling.

        Returns
        -------
        ndarray
            Best parameter found for each row in X.
        """
        p = np.tile(np.linspace(*p_range, sample_density), (len(X), 1))
        feed_dict = {self._x_in: X, self._p_in: p}
        if use_head >= 0:
            feed_dict[self._h_sel] = use_head
        best_p = self._sess.run(self._best_p, feed_dict=feed_dict)
        return p[np.arange(len(p)), best_p]

    def estimate_best_p(self, X, p_range=(0, 1), sample_density=100):
        """Sample a range of parameters and return an estimate for the best.

        The best guess is estimated from the mean of all heads' predictions.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p_range : 2-tuple, optional
            The range used to explore the parameter space.
        sample_density : int, optional
            Number of uniform samples to explore within p_range.

        Returns
        -------
        ndarray
            Best parameter found for each row in X.
        """
        p = np.tile(np.linspace(*p_range, sample_density), (len(X), 1))
        best_p = self._sess.run(self._mbest_p, feed_dict={self._x_in: X, self._p_in: p})
        return p[np.arange(len(p)), best_p]

    def estimate_reward_curve(self, X, p_range=(0, 1), sample_density=100, multi_output=False):
        """Sample a range of parameters and return an estimated reward curve.

        The curve is averaged over heads, so this is a central estimation.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p_range : 2-tuple, optional
            Lower and upper bound of the parameter space.
        sample_density : int, optional
            Number of uniformly spaced samples to explore.
        multi_output : boolean, optional
            Whether to return a single estimate or several (one for each head).

        Returns
        -------
        space : ndarray
            The parameter space.
        curve : ndarray
            The estimated rewards for each sampled parameter.
        """
        s = np.linspace(*p_range, sample_density)
        p = np.tile(s, (len(X), 1))
        if multi_output:
            return s, self._sess.run(self._h_out, feed_dict={self._x_in: X, self._p_in: p})
        else:
            return s, self._sess.run(self._h_mean, feed_dict={self._x_in: X, self._p_in: p})

    def sample_reward(self, X, p, use_head=-1):
        """Sample reward estimates for each data sample and each provided parameter.

        Note
        ----
        For now, the same head is sampled for every row.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p : ndarray
            The parameters to run.
        use_head : int, optional
            Force use of a specific head for sampling.

        Returns
        -------
        ndarray
            The sampled rewards.
        """
        X = np.asanyarray(X, dtype=np.float32)
        p = np.asanyarray(p, dtype=np.float32)

        if p.ndim < 2:
            p = p.reshape(-1, 1)  # try to promote to rank 2
            assert len(p) == len(X), "Wrong shape for parameters vector."

        feed_dict = {self._x_in: X, self._p_in: p}
        if use_head >= 0:
            feed_dict[self._h_sel] = use_head

        r = self._sess.run(self._y_out, feed_dict=feed_dict)
        return r.squeeze()

    def estimate_reward(self, X, p):
        """Estimate rewards for each data sample and each provided parameter.

        The curve is averaged over heads, so this is a central estimation.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p : ndarray
            The parameters to run.

        Returns
        -------
        ndarray
            The estimated rewards.
        """
        X = np.asanyarray(X, dtype=np.float32)
        p = np.asanyarray(p, dtype=np.float32)

        if p.ndim < 2:
            p = p.reshape(-1, 1)  # try to promote to rank 2
            assert len(p) == len(X), "Wrong shape for parameters vector."

        r = self._sess.run(self._h_mean, feed_dict={self._x_in: X, self._p_in: p})
        return r.squeeze()

    def evaluate_p(self, X, best_p, p_range=(0, 1), sample_density=100):
        """Evaluate the correctness of the estimated best parameters.

        The estimates are averaged over heads, so this evaluates the central measure.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        best_p : ndarray
            The parameters to run.
        p_range : 2-tuple, optional
            The range used to explore the parameter space.
        sample_density : int, optional
            Number of uniform samples to explore within p_range.

        Returns
        -------
        float
            The mean squared error for the estimated best parameters.
        """
        return np.mean((best_p - self.estimate_best_p(X, p_range, sample_density)) ** 2)

    def evaluate(self, X, p, y, eps=1e-5):
        """Evaluate the correctness of the estimated rewards.

        The estimates are averaged over heads, so this evaluates the central measure.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        p : ndarray
            The parameters tried (may be more than 1 for each X row).
        y : ndarray
            The true rewards (same shape as p).
        eps : float, optional
            Small positive number to compute cross entropy with.
            Only meaningful when loss type is 'ce'.

        Returns
        -------
        float
            The loss over the estimated and true rewards.
        """
        if self.loss_type == 'ce':
            y_pred = np.clip(self.estimate_reward(X, p), eps, 1-eps)
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        elif self.loss_type == 'mse':
            return np.mean((y - self.estimate_reward(X, p)) ** 2)
