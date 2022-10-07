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
"""Tests for unsupervised embedding losses."""
import os

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn.runner import orchestration
from tensorflow_gnn.runner.tasks import unsupervised

SCHEMA = """
node_sets {
  key: "node"
  value {
    features {
      key: "%s"
      value {
        dtype: DT_FLOAT
        shape { dim { size: 4 } }
      }
    }
  }
}
edge_sets {
  key: "edge"
  value {
    source: "node"
    target: "node"
  }
}
""" % tfgnn.HIDDEN_STATE


class _ContrastiveTestMixIn:

  gtspec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.parse_schema(SCHEMA))
  hidden_dim = 8
  message_dim = 16
  task = None

  def build_model(self):
    graph = inputs = tf.keras.layers.Input(type_spec=self.gtspec)

    for _ in range(2):  # Message pass twice
      values = tfgnn.broadcast_node_to_edges(
          graph, "edge", tfgnn.TARGET, feature_name=tfgnn.HIDDEN_STATE)
      messages = tf.keras.layers.Dense(self.message_dim)(values)

      pooled = tfgnn.pool_edges_to_node(
          graph,
          "edge",
          tfgnn.SOURCE,
          reduce_type="sum",
          feature_value=messages)
      h_old = graph.node_sets["node"].features[tfgnn.HIDDEN_STATE]

      h_next = tf.keras.layers.Concatenate()((pooled, h_old))
      h_next = tf.keras.layers.Dense(self.hidden_dim)(h_next)

      graph = graph.replace_features(
          node_sets={"node": {
              tfgnn.HIDDEN_STATE: h_next
          }})

    return tf.keras.Model(inputs=inputs, outputs=graph)

  # Used for both test_fit and downstream test_training
  def fit(self):
    gt = tfgnn.random_graph_tensor(self.gtspec)
    ds = tf.data.Dataset.from_tensors(gt).repeat(8)
    ds = ds.batch(2).map(tfgnn.GraphTensor.merge_batch_to_components)

    model = self.task.adapt(self.build_model())
    model.compile()

    def get_loss():
      values = model.evaluate(ds)
      return dict(zip(model.metrics_names, values))["loss"]

    before = get_loss()
    model.fit(ds)
    after = get_loss()
    return before, after

  def test_fit(self):
    self.fit()

  def test_adapt(self):
    model = self.build_model()
    adapted = self.task.adapt(model)

    gt = tfgnn.random_graph_tensor(self.gtspec)
    # Output should be x_clean (i.e., root node representation)
    expected = tfgnn.keras.layers.ReadoutFirstNode(
        node_set_name="node", feature_name=tfgnn.HIDDEN_STATE)(
            model(gt))
    actual = adapted(gt)

    self.assertAllClose(actual, expected)

  def test_protocol(self):
    self.assertIsInstance(self.task.__class__, orchestration.Task)

  def load_model(self) -> tf.keras.Model:
    model = self.task.adapt(self.build_model())
    export_dir = os.path.join(self.get_temp_dir(), self.__class__.__name__)
    model.save(export_dir, include_optimizer=False)
    # Test both loading ways.
    model = tf.saved_model.load(export_dir)
    return tf.keras.models.load_model(export_dir)


class DeepGraphInfomaxTest(tf.test.TestCase, _ContrastiveTestMixIn):

  task = unsupervised.DeepGraphInfomax("node", seed=8191)

  def test_model_saving(self):
    model = self.load_model()
    self.assertIsInstance(
        model.get_layer(index=-1), unsupervised.AddLossDeepGraphInfomax)

  def test_training(self):
    before, after = self.fit()
    self.assertAllClose(before, 250.42036, rtol=1e-04, atol=1e-04)
    self.assertAllClose(after, 13.18533, rtol=1e-04, atol=1e-04)


class BarlowTwinsTest(tf.test.TestCase, _ContrastiveTestMixIn):

  task = unsupervised.BarlowTwins("node", seed=1337)

  def test_model_saving(self):
    model = self.load_model()
    self.assertIsInstance(
        model.get_layer(index=-1), unsupervised.AddLossBarlowTwins)

  def test_training(self):
    before, after = self.fit()
    self.assertAllClose(before, 8, rtol=1e-04, atol=1e-04)
    self.assertAllClose(after, 8, rtol=1e-04, atol=1e-04)


class VICRegTest(tf.test.TestCase, _ContrastiveTestMixIn):

  task = unsupervised.VICReg("node", seed=42)

  def test_model_saving(self):
    model = self.load_model()
    self.assertIsInstance(model.get_layer(index=-1), unsupervised.AddLossVICReg)

  def test_training(self):
    before, after = self.fit()
    self.assertAllClose(before, 89.5600, rtol=1e-04, atol=1e-04)
    self.assertAllClose(after, 64.8361, rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
  tf.test.main()
