# Copyright 2022 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unsupervised tasks."""
import abc
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow_gnn as tfgnn


class _ConstrastiveLossTask(abc.ABC):
  """Base class for unsupervised contrastive representation learning tasks.

  The default `adapt` method implementation shuffles feature across batch
  examples to create positive and negative activations (b/255491406). There are
  multiple ways proposed in the literature to learn representations based on the
  activations.

  Any subclass must implement `contrast` method, which re-adapts the input
  `tf.keras.Model` to prepare task-specific outputs and/or adds necessary
  contrastive losses to the `tf.keras.Model` itself.

  If the loss involves labels for each example, subclasses should leverage
  `losses` and `metrics` methods to specify task's losses. When the loss only
  involves model outputs, `contrast` should inject the loss into the
  `tf.keras.Model` directly.

  Any model-specific preprocessing should be implemented in the `preprocessing`.
  """

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               seed: Optional[int] = None):
    self._state_name = state_name
    self._node_set_name = node_set_name
    self._seed = seed

  def adapt(self, model: tf.keras.Model) -> tf.keras.Model:
    """Adapt a `tf.keras.Model` for use with various contrastive losses.

    The input `tf.keras.Model` must have a single `GraphTensor` input and a
    single `GraphTensor` output.

    Args:
      model: A `tf.keras.Model` to be adapted.

    Returns:
      A `tf.keras.Model` with output logits for contrastive loss: a positive
      logit and a negative logit for each example in a batch.
    """
    if not tfgnn.is_graph_tensor(model.input):
      raise ValueError(f"Expected a GraphTensor, received {model.input}")

    if not tfgnn.is_graph_tensor(model.output):
      raise ValueError(f"Expected a GraphTensor, received {model.output}")

    # Clean representations: readout
    x_clean = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name, feature_name=self._state_name)(
            model.output)

    # Corrupted representations: shuffling, model application and readout
    shuffled = tfgnn.shuffle_features_globally(model.input)
    x_corrupted = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name=self._node_set_name, feature_name=self._state_name)(
            model(shuffled))

    return self.contrast_outputs(
        tf.keras.Model(model.input, [x_clean, x_corrupted]))

  @abc.abstractmethod
  def contrast_outputs(self, model: tf.keras.Model) -> tf.keras.Model:
    raise NotImplementedError()

  def preprocess(self, gt: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    """Returns the input GraphTensor."""
    return gt

  def losses(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Returns an empty losses tuple.

    Loss signatures are according to `tf.keras.losses.Loss,` here: no losses
    are returned because they have been added to the model via
    `tf.keras.Layer.add_loss.`
    """
    return tuple()

  def metrics(self) -> Sequence[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Returns an empty metrics tuple.

    Metric signatures are according to `tf.keras.metrics.Metric,` here: no
    metrics are returned because they have been added to the model via
    `tf.keras.Layer.add_metric.`
    """
    return tuple()


class BarlowTwins(_ConstrastiveLossTask):
  """A Task to minimize the BarlowTwins loss.

  BarlowTwins is an unsupervised loss that attempts to contrast representations
  by discriminating between positive examples (any input `GraphTensor`) and
  negative examples (an input `GraphTensor` but with corrupted features: this
  implementation shuffles GraphTensor features across all the nodes (and edges)
  of all input examples in a training batch (separately for each node/edge set).
  BarlowTwins loss optimizes the covariance matrix between clean and corrupted
  representations.

  This task re-adapts a `tf.keras.Model` with single `GraphTensor` input and a
  single `GraphTensor` output. The outputs of the re-adapted model are positive
  and negative activations for clean and corrupted `GraphTensor`s, and the
  BarlowTwins is added to the `tf.keras.Model`'s losses.

  For more information, see: https://arxiv.org/abs/2103.03230.
  """

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               seed: Optional[int] = None,
               lambda_: float = 0,
               normalize_batch: bool = True):
    super(BarlowTwins, self).__init__(
        node_set_name, state_name=state_name, seed=seed)
    self._lambda = lambda_
    self._normalize_batch = normalize_batch

  def contrast_outputs(self, model: tf.keras.Model) -> tf.keras.Model:
    """Readapt a `tf.keras.Model` for BarlowTwins.

    The input `tf.keras.Model` must have a single `GraphTensor` input and a
    single `GraphTensor` output.

    Args:
      model: A `tf.keras.Model` to be adapted.

    Returns:
      A `tf.keras.Model` with output logits for BarlowTwins: a stacked positive
      logit and a negative logit for each example in a batch.
    """
    x_clean, x_corrupted = tf.unstack(model.outputs, 2)
    return tf.keras.Model(
        model.input,
        AddLossBarlowTwins(
            lambda_=self._lambda,
            normalize_batch=self._normalize_batch)((x_clean, x_corrupted)))


class VICReg(_ConstrastiveLossTask):
  """A task to minimize VICReg loss.

  VICReg is an unsupervised loss that attempts to contrast representations
  by discriminating between positive examples (any input `GraphTensor`) and
  negative examples (an input `GraphTensor` but with corrupted features: this
  implementation shuffles GraphTensor features across all the nodes (and edges)
  of all input examples in a training batch (separately for each node/edge set).

  This task re-adapts a `tf.keras.Model` with single `GraphTensor` input and a
  single `GraphTensor` output. The outputs of the re-adapted model are positive
  and negative activations for clean and corrupted `GraphTensor`s, and the
  VICReg is added to the `tf.keras.Model`'s losses.

  For more information, see: https://arxiv.org/abs/2105.04906.
  """

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               seed: Optional[int] = None,
               sim_weight=25,
               var_weight=25,
               cov_weight=1):
    super(VICReg, self).__init__(
        node_set_name, state_name=state_name, seed=seed)
    self._sim_weight = sim_weight
    self._var_weight = var_weight
    self._cov_weight = cov_weight

  def contrast_outputs(self, model: tf.keras.Model) -> tf.keras.Model:
    """Readapt a `tf.keras.Model` for VicReg.

    The input `tf.keras.Model` must have a single `GraphTensor` input and a
    single `GraphTensor` output.

    Args:
      model: A `tf.keras.Model` to be adapted.

    Returns:
      A `tf.keras.Model` with output logits for VicReg: a stacked positive logit
      and a negative logit for each example in a batch.
    """
    x_clean, x_corrupted = tf.unstack(model.outputs, 2)
    return tf.keras.Model(
        model.input,
        AddLossVICReg(self._sim_weight, self._var_weight,
                      self._cov_weight)((x_clean, x_corrupted)))


class DeepGraphInfomax(_ConstrastiveLossTask):
  """A Task for training with the Deep Graph Infomax loss.

  Deep Graph Infomax is an unsupervised loss that attempts to learn a bilinear
  layer capable of discriminating between positive examples (any input
  `GraphTensor`) and negative examples (an input `GraphTensor` but with shuffled
  features: this implementation shuffles features across the components of a
  scalar `GraphTensor`).

  Deep Graph Infomax is particularly useful in unsupervised tasks that wish to
  learn latent representations informed primarily by a nodes neighborhood
  attributes (vs. its structure).

  This task can adapt a `tf.keras.Model` with a single, scalar `GraphTensor`
  input and a single, scalar `GraphTensor`  output. The adapted `tf.keras.Model`
  head has--as its output--any latent, root node (according to `node_set_name`
  and `state_name`) represenations. The unsupervised loss is added to the model
  by adding a Layer that calls `tf.keras.Layer.add_loss().`

  For more information, see: https://arxiv.org/abs/1809.10341.
  """

  def __init__(self,
               node_set_name: str,
               *,
               state_name: str = tfgnn.HIDDEN_STATE,
               seed: Optional[int] = None):
    super(DeepGraphInfomax, self).__init__(
        node_set_name, state_name=state_name, seed=seed)

  def contrast_outputs(self, model: tf.keras.Model) -> tf.keras.Model:
    """Readapt a `tf.keras.Model` for Deep Graph Infomax.

    The input `tf.keras.Model` must have a single `GraphTensor` input and a
    single `GraphTensor` output.

    Args:
      model: A `tf.keras.Model` to be adapted.

    Returns:
      A `tf.keras.Model` with output logits for Deep Graph Infomax: a positive
      logit and a negative logit for each example in a batch.
    """
    x_clean, x_corrupted = tf.unstack(model.outputs, 2)
    return tf.keras.Model(
        model.input,
        AddLossDeepGraphInfomax(x_clean.get_shape()[-1])(
            (x_clean, x_corrupted)))


@tf.keras.utils.register_keras_serializable(package="GNN")
class AddLossDeepGraphInfomax(tf.keras.layers.Layer):
  """A bilinear layer with losses and metrics for Deep Graph Infomax."""

  def __init__(self, units: int, **kwargs):
    """Builds the bilinear layer weights.

    Args:
      units: Units for the bilinear layer.
      **kwargs: Extra arguments needed for serialization.
    """
    super().__init__(**kwargs)
    self._bilinear = tf.keras.layers.Dense(units, use_bias=False)

  def get_config(self) -> Mapping[Any, Any]:
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.
    """
    return dict(units=self._bilinear.units, **super().get_config())

  def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Returns clean representations after adding a Deep Graph Infomax loss.

    Clean representations are the unmanipulated, original model output.

    Args:
      inputs: A tuple of (clean, corrupted) representations for Deep Graph
        Infomax.

    Returns:
      The clean representations: the first item of the `inputs` tuple.
    """
    y_clean, y_corrupted = inputs
    # Summary
    summary = tf.math.reduce_mean(y_clean, axis=0, keepdims=True)
    # Clean losses and metrics
    logits_clean = tf.matmul(y_clean, self._bilinear(summary), transpose_b=True)
    self.add_loss(
        tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            name="binary_crossentropy_clean")(tf.ones_like(logits_clean),
                                              logits_clean))
    self.add_metric(
        tf.keras.metrics.binary_crossentropy(
            tf.ones_like(logits_clean), logits_clean, from_logits=True),
        name="binary_crossentropy_clean")
    self.add_metric(
        tf.keras.metrics.binary_accuracy(
            tf.ones_like(logits_clean), logits_clean),
        name="binary_accuracy_clean")
    # Corrupted losses and metrics
    logits_corrupted = tf.matmul(
        y_corrupted, self._bilinear(summary), transpose_b=True)
    self.add_loss(
        tf.keras.losses.BinaryCrossentropy(
            from_logits=True, name="binary_crossentropy_corrupted")(
                tf.zeros_like(logits_corrupted), logits_corrupted))
    self.add_metric(
        tf.keras.metrics.binary_crossentropy(
            tf.zeros_like(logits_corrupted), logits_corrupted,
            from_logits=True),
        name="binary_crossentropy_corrupted")
    self.add_metric(
        tf.keras.metrics.binary_accuracy(
            tf.zeros_like(logits_corrupted), logits_corrupted),
        name="binary_accuracy_corrupted")
    return y_clean


@tf.keras.utils.register_keras_serializable(package="GNN")
class AddLossBarlowTwins(tf.keras.layers.Layer):
  """Barlow Twins loss implementation as a `tf.keras.layers.Layer`.

  This layer adds BarlowTwins loss and metric into the model, because Keras does
  not allow unsupervised losses to be added natively.

  Barlow Twins loss optimizes cross-correlation between batch-normalized clean
  and corrupted representations. For more details, see the orinial paper:
  https://arxiv.org/abs/2103.03230

  Attributes:
    lambda_: Parameter lambda of the BarlowTwins model. If 0 (default), uses `1
      / feature_dim`.
    normalize_batch: If `True`, normalizes representations across the batch
      dimension.
  """

  def __init__(self, *, lambda_=0, normalize_batch=True, **kwargs):
    super(AddLossBarlowTwins, self).__init__(**kwargs)
    self._lambda = lambda_
    self._normalize_batch = normalize_batch

  def call(self, inputs):
    representations_clean, representations_corrupted = tf.unstack(inputs, 2)
    loss = _barlow_twins_loss(representations_clean, representations_corrupted,
                              self._lambda, self._normalize_batch)
    self.add_loss(loss)
    self.add_metric(loss, "Loss")
    return representations_clean

  def get_config(self) -> Mapping[Any, Any]:
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.
    """
    return dict(
        lambda_=self._lambda,
        normalize_batch=self._normalize_batch,
        **super().get_config())


@tf.keras.utils.register_keras_serializable(package="GNN")
class AddLossVICReg(tf.keras.layers.Layer):
  """VICReg loss implementation as a `tf.keras.layers.Layer`.

  This layer adds VICReg loss and metric into the model, because Keras does not
  allow unsupervised losses to be added natively.

  VICReg loss consists of three components: Variance and Covariance losses that
  are applied to both clean and corrupted representations separately, and an
  Invariance loss that computes the similarity between representations.
  For more details, see the orinial paper: https://arxiv.org/abs/2105.04906

  Attributes:
    sim_weight: Weight of the invariance (similarity) loss component of the
      VICReg loss.
    var_weight: Weight of the variance loss component of the VICReg loss.
    cov_weight: Weight of the covariance loss component of the VICReg loss.
  """

  def __init__(self, sim_weight=25, var_weight=25, cov_weight=1, **kwargs):
    super(AddLossVICReg, self).__init__(**kwargs)
    self._sim_weight = sim_weight
    self._var_weight = var_weight
    self._cov_weight = cov_weight

  def call(self, inputs):
    representations_clean, representations_corrupted = tf.unstack(inputs, 2)
    total_loss = _vicreg_loss(representations_clean, representations_corrupted,
                              self._sim_weight, self._var_weight,
                              self._cov_weight)
    self.add_loss(total_loss)
    self.add_metric(total_loss, "VICRegLoss")
    return representations_clean

  def get_config(self) -> Mapping[Any, Any]:
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable) containing the
    configuration of a layer. The same layer can be reinstantiated later
    (without its trained weights) from this configuration.
    """
    return dict(
        sim_weight=self._sim_weight,
        var_weight=self._var_weight,
        cov_weight=self._cov_weight,
        **super().get_config())


def _variance_loss(representations: tf.Tensor, eps: float = 1e-4) -> tf.Tensor:
  """Variance loss component of VicReg loss.

  Computes truncated per-dimension standard deviation of input representations.

  Args:
    representations: Input representations.
    eps: Epsilon for the standard deviation computation.

  Returns:
    Loss value as scalar `tf.Tensor`.
  """
  std = tf.math.sqrt(tf.math.reduce_variance(representations, axis=0) + eps)
  return tf.reduce_mean(tf.nn.relu(1 - std))


def _covariance_loss(representations: tf.Tensor) -> tf.Tensor:
  """Covariance loss component of VicReg loss.

  Computes normalized square of the off-diagonal elements of the covariance
  matrix of input representations.

  Args:
    representations: Input representations.

  Returns:
    Loss value as scalar `tf.Tensor`.
  """
  batch_size, feature_dim = tf.unstack(
      tf.cast(tf.shape(representations), tf.float32))
  representations = _normalize(representations, scale=False)
  covariance = tf.matmul(
      representations, representations, transpose_a=True) / batch_size
  covariance = tf.pow(covariance, 2)
  return tf.reduce_sum(
      tf.linalg.set_diag(covariance, tf.zeros(
          tf.shape(representations)[1]))) / feature_dim


def _normalize(representations: tf.Tensor, *, scale=True) -> tf.Tensor:
  """Standardizes the representations across the first (batch) dimension.

  Args:
    representations: Representations to normalize.
    scale: Whether to scale representations by the standard deviation. If
      `False`, simply remove the mean from `features`. Default: `True`.

  Returns:
    Normalized representations as `tf.Tensor`.
  """
  representations_mean = tf.reduce_mean(representations, axis=0)
  if scale:
    representations_std = tf.math.reduce_std(representations, axis=0)
    return tf.math.divide_no_nan(representations - representations_mean,
                                 representations_std)
  else:
    return representations - representations_mean


def _vicreg_loss(representations_clean: tf.Tensor,
                 representations_corrupted: tf.Tensor,
                 sim_weight: Union[tf.Tensor, float] = 25,
                 var_weight: Union[tf.Tensor, float] = 25,
                 cov_weight: Union[tf.Tensor, float] = 1) -> tf.Tensor:
  """VICReg loss implementation.

  Implements VICReg loss from the paper https://arxiv.org/abs/2105.04906

  Args:
    representations_clean: Representations from the clean view of the data.
    representations_corrupted: Representations from the corrupted view of the
      data.
    sim_weight: Weight of the invariance (similarity) loss component of the
      VICReg loss.
    var_weight: Weight of the variance loss component of the VICReg loss.
    cov_weight: Weight of the covariance loss component of the VICReg loss.

  Returns:
    VICReg loss value as `tf.Tensor`.
  """
  losses = []
  if tf.get_static_value(sim_weight) != 0.0:
    losses.append(sim_weight * tf.keras.losses.MeanSquaredError()(
        representations_clean, representations_corrupted))
  if tf.get_static_value(var_weight) != 0.0:
    losses.append(var_weight * _variance_loss(representations_clean))
    losses.append(var_weight * _variance_loss(representations_corrupted))
  if tf.get_static_value(cov_weight) != 0.0:
    losses.append(cov_weight * _covariance_loss(representations_clean))
    losses.append(cov_weight * _covariance_loss(representations_corrupted))
  return tf.add_n(losses)


def _barlow_twins_loss(representations_clean: tf.Tensor,
                       representations_corrupted: tf.Tensor,
                       lambda_: Optional[Union[tf.Tensor, float]] = None,
                       normalize_batch: bool = True) -> tf.Tensor:
  """Barlow Twins loss implementation.

  Implements BarlowTwins loss from the paper https://arxiv.org/abs/2103.03230

  Args:
    representations_clean: Representations from the clean view of the data.
    representations_corrupted: Representations from the corrupted view of the
      data.
    lambda_: Parameter lambda of the BarlowTwins model. If None (default), uses
      `1 / feature_dim`.
    normalize_batch: If `True` (default), normalizes representations per-batch.

  Returns:
    BarlowTwins loss value as `tf.Tensor`.
  """
  if normalize_batch:
    representations_clean = _normalize(representations_clean)
    representations_corrupted = _normalize(representations_corrupted)
  batch_size, feature_dim = tf.unstack(
      tf.cast(tf.shape(representations_clean), tf.float32))
  lambda_ = 1 / feature_dim if lambda_ is None else lambda_
  correlation = tf.linalg.matmul(
      representations_clean, representations_corrupted,
      transpose_a=True) / batch_size

  loss_matrix = tf.pow(correlation - tf.eye(feature_dim), 2)
  loss_diagonal_sum = tf.linalg.trace(loss_matrix)
  loss_sum = tf.reduce_sum(loss_matrix)
  return (1. - lambda_) * loss_diagonal_sum + lambda_ * loss_sum
