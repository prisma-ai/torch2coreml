import numpy as np
import numpy.testing as npt
import unittest
import os
import sys
import torch.legacy.nn as nn

from PIL import Image
from _test_utils import _INPUT_SHAPE

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../torch2coreml/"
)


class TorchConverterTest(unittest.TestCase):
    def setUp(self):
        self.input = np.random.ranf(_INPUT_SHAPE)
        self.model = nn.Sequential()
        self.model.add(nn.MulConstant(1.0))

    def test_image_input(self):
        from _torch_converter import convert
        coreml_model = convert(
            self.model,
            [self.input.shape],
            input_names=['image'],
            image_input_names=['image'],
            preprocessing_args={
                'is_bgr': False,
                'red_bias': 0.0,
                'green_bias': 0.0,
                'blue_bias': 0.0,
                'image_scale': 0.5
            }
        )

        input_array = (np.random.rand(224, 224, 3) * 255).astype('uint8')
        input_image = Image.fromarray(input_array).convert('RGBA')
        output_array = coreml_model.predict({"image": input_image})["output"]
        output_array = output_array.transpose((1, 2, 0))
        npt.assert_array_equal(output_array, input_array * 0.5)

    def test_image_output(self):
        from _torch_converter import convert
        coreml_model = convert(
            self.model,
            [self.input.shape],
            input_names=['image'],
            output_names=['output'],
            image_input_names=['image'],
            preprocessing_args={
                'is_bgr': False,
                'red_bias': -10.0,
                'green_bias': -20.0,
                'blue_bias': -30.0,
                'image_scale': 1.0
            },
            image_output_names=['output'],
            deprocessing_args={
                'is_bgr': False,
                'red_bias': 10.0,
                'green_bias': 20.0,
                'blue_bias': 30.0,
                'image_scale': 1.0
            }
        )

        input_array = (np.random.rand(224, 224, 3) * 255).astype('uint8')
        input_image = Image.fromarray(input_array).convert('RGBA')

        output_image = coreml_model.predict({"image": input_image})["output"]
        output_array = np.array(output_image.convert('RGB'))

        npt.assert_array_equal(output_array, input_array)

    def test_image_deprocess_scale(self):
        from _torch_converter import convert
        coreml_model = convert(
            self.model,
            [self.input.shape],
            input_names=['image'],
            output_names=['output'],
            image_input_names=['image'],
            preprocessing_args={
                'is_bgr': False,
                'red_bias': 0.0,
                'green_bias': 0.0,
                'blue_bias': 0.0,
                'image_scale': 0.5
            },
            image_output_names=['output'],
            deprocessing_args={
                'is_bgr': False,
                'red_bias': 0.0,
                'green_bias': 0.0,
                'blue_bias': 0.0,
                'image_scale': 2.0
            }
        )

        input_array = (np.random.rand(224, 224, 3) * 255).astype('uint8')
        input_image = Image.fromarray(input_array).convert('RGBA')

        output_image = coreml_model.predict({"image": input_image})["output"]
        output_array = np.array(output_image.convert('RGB'))

        npt.assert_array_equal(output_array, input_array)

    def test_classifier(self):
        from _torch_converter import convert

        class_labels = ['class1', 'class2', 'class3']

        coreml_model = convert(
            self.model,
            [(3,)],
            mode='classifier',
            class_labels=class_labels,
            predicted_feature_name='class'
        )

        input_array = [0.0, 1.0, 0.0]

        predicted_class = coreml_model.predict({'input': input_array})['class']
        self.assertEqual(predicted_class, class_labels[1])
