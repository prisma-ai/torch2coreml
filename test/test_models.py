import numpy as np
import unittest
import sys
import os
import subprocess
import tempfile

import torch
from torch.utils.serialization import load_lua

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../torch2coreml/"
)


class ResNetTest(unittest.TestCase):
    def setUp(self):
        _, model_path = tempfile.mkstemp()
        self.model_path = model_path

        models_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'models'
        )

        returncode = subprocess.call(
            [
                'th',
                os.path.join(models_dir, 'create_resnet_model.lua'),
                '-depth', '18',
                '-path', self.model_path
            ],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        self.assertEqual(returncode, 0)

    def tearDown(self):
        os.remove(self.model_path)

    def test_resnet18(self):
        torch_net = load_lua(self.model_path)
        torch_net.evaluate()

        input_0 = np.random.ranf((3, 224, 224))

        input_tensor = torch.from_numpy(np.asarray([input_0])).float()

        torch_output = torch_net.forward(input_tensor).numpy().flatten()

        from _torch_converter import convert
        coreml_net = convert(
            self.model_path,
            (3, 224, 224)
        )
        coreml_output = coreml_net.predict({'input': input_0})['output']

        corrcoef = np.corrcoef(coreml_output, torch_output).flatten()[1]
        self.assertAlmostEqual(corrcoef, 1.0, delta=1e-4)
