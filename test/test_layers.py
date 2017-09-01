import numpy as np
import numpy.testing as npt
import unittest
import sys
import os
import tempfile

import torch
from torch.utils.serialization import load_lua

from _test_utils import _generate_single_layer_torch_model, _INPUT_SHAPE

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../torch2coreml/"
)


class SingleLayerTest(unittest.TestCase):
    def setUp(self):
        _, model_path = tempfile.mkstemp()
        self.model_path = model_path
        self.input = np.random.ranf(_INPUT_SHAPE)
        self.torch_batch_mode = True

    def tearDown(self):
        os.remove(self.model_path)

    def _forward_torch(self):
        torch_model = load_lua(self.model_path)
        if self.torch_batch_mode:
            input_tensor = torch.from_numpy(np.asarray([self.input])).float()
            return torch_model.forward(input_tensor).numpy()[0]
        else:
            input_tensor = torch.from_numpy(self.input).float()
            return torch_model.forward(input_tensor).numpy()

    def _forward_coreml(self):
        from _torch_converter import convert
        coreml_model = convert(self.model_path, self.input.shape)
        return coreml_model.predict({'input': self.input})['output']

    def _assert_outputs(self, torch_output, coreml_output):
        self.assertEqual(torch_output.shape, coreml_output.shape)
        npt.assert_almost_equal(coreml_output, torch_output, decimal=2)
        corrcoef = np.corrcoef(coreml_output.flatten(),
                               torch_output.flatten()).flatten()[1]
        self.assertAlmostEqual(corrcoef, 1.0, delta=1e-4)

    def _test_single_layer(self, layer):
        _generate_single_layer_torch_model(
            layer, self.input.shape, self.model_path
        )
        torch_output = self._forward_torch()
        coreml_output = self._forward_coreml()
        self._assert_outputs(torch_output, coreml_output)

    def test_elu(self):
        self._test_single_layer('nn.ELU()')

    def test_relu(self):
        self._test_single_layer('nn.ReLU()')

    def test_softmax(self):
        self._test_single_layer('nn.SoftMax()')
        self._test_single_layer('nn.SpatialSoftMax()')

    def test_convolution(self):
        self._test_single_layer('nn.SpatialConvolution(3,64,7,7,2,2,3,3)')

    def test_max_pooling(self):
        self._test_single_layer('nn.SpatialMaxPooling(3,3,1,1,1,1)')

    def test_avg_pooling(self):
        self._test_single_layer('nn.SpatialAveragePooling(5,5,1,1,2,2)')

    def test_linear(self):
        self.input = self.input.flatten()
        input_size = self.input.shape[0]
        self._test_single_layer('nn.Linear({input_size}, 3, true)'.format(
            input_size=input_size
        ))

    def test_tanh(self):
        self._test_single_layer('nn.Tanh()')

    def test_mul_constant(self):
        self._test_single_layer('nn.MulConstant(3.0)')

    def test_zero_padding(self):
        self._test_single_layer('nn.SpatialZeroPadding(1, 2, 3, 4)')
        self._test_single_layer('nn.SpatialZeroPadding(-2, -2, -2, -2)')

    def test_full_convolution(self):
        self._test_single_layer(
            'nn.SpatialFullConvolution(3,1,7,7,5,5,2,2,2,2)'
        )

    def test_batch_norm(self):
        self._test_single_layer('nn.SpatialBatchNormalization(3)')

    def test_narrow(self):
        self.torch_batch_mode = False
        self._test_single_layer('nn.Narrow(1, 1, 1)')

    def test_reflection_padding(self):
        self._test_single_layer('nn.SpatialReflectionPadding(1, 2, 3, 4)')

    def test_upsample_nearest(self):
        self._test_single_layer('nn.SpatialUpSamplingNearest(2)')
