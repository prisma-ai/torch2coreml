import numpy as np
import torch

from torch.utils.serialization import load_lua

from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import MLModel, datatypes

import _layers

from _utils import _gen_layer_name
from _utils import _convert_multiarray_output_to_image


_DEPROCESS_LAYER_NAME = 'deprocess_image'


def _infer_torch_output_shape(torch_model, input_shape):
    """
    Forward torch model to infer output shape
    """
    try:
        input_tensor = torch.rand(*input_shape).float()
        output_shape = torch_model.forward(input_tensor).numpy().shape
        return output_shape
    except:
        # try batch mode
        input_tensor = torch.rand(1, *input_shape).float()
        output_shape = torch_model.forward(input_tensor).numpy().shape[1:]
        return output_shape


def _set_deprocessing(is_grayscale,
                      builder,
                      deprocessing_args,
                      input_name,
                      output_name):
    is_bgr = deprocessing_args.get('is_bgr', False)

    _convert_multiarray_output_to_image(
        builder.spec, output_name, is_bgr=is_bgr
    )

    image_scale = deprocessing_args.get('image_scale', 1.0)

    if is_grayscale:
        gray_bias = deprocessing_args.get('gray_bias', 0.0)
        W = np.array([image_scale])
        b = np.array([gray_bias])
    else:
        W = np.array([image_scale, image_scale, image_scale])

        red_bias = deprocessing_args.get('red_bias', 0.0)
        green_bias = deprocessing_args.get('green_bias', 0.0)
        blue_bias = deprocessing_args.get('blue_bias', 0.0)

        if not is_bgr:
            b = np.array([
                red_bias,
                green_bias,
                blue_bias,
            ])
        else:
            b = np.array([
                blue_bias,
                green_bias,
                red_bias,
            ])

    builder.add_scale(
        name=_DEPROCESS_LAYER_NAME,
        W=W,
        b=b,
        has_bias=True,
        input_name=input_name,
        output_name=output_name
    )


def convert(model,
            input_shape,
            input_name='input',
            output_name='output',
            mode=None,
            is_image_input=False,
            preprocessing_args={},
            is_image_output=False,
            deprocessing_args={},
            class_labels=None,
            predicted_feature_name='classLabel'):
    """
    Convert Torch7 model to CoreML.

    Parameters
    ----------
    model: Torch7 model (loaded with PyTorch) | str
        A trained Torch7 model loaded in python using PyTorch or path to file
        with model (*.t7).

    input_shape: tuple
        Shape of the input tensor.

    mode: str ('classifier', 'regressor' or None)
        Mode of the converted coreml model:
        'classifier', a NeuralNetworkClassifier spec will be constructed.
        'regressor', a NeuralNetworkRegressor spec will be constructed.

    preprocessing_args: dict
        'is_bgr', 'red_bias', 'green_bias', 'blue_bias', 'gray_bias',
        'image_scale' keys with the same meaning as
        https://apple.github.io/coremltools/generated/coremltools.models.neural_network.html#coremltools.models.neural_network.NeuralNetworkBuilder.set_pre_processing_parameters

    deprocessing_args: dict
        Same as 'preprocessing_args' but for deprocessing.

    class_labels: A string or list of strings.
        As a string it represents the name of the file which contains
        the classification labels (one per line).
        As a list of strings it represents a list of categories that map
        the index of the output of a neural network to labels in a classifier.

    predicted_feature_name: str
        Name of the output feature for the class labels exposed in the Core ML
        model (applies to classifiers only). Defaults to 'classLabel'

    Returns
    -------
    model: A coreml model.
    """
    _gen_layer_name.called = 0

    if isinstance(model, basestring):
        torch_model = load_lua(model)
    elif isinstance(model, torch.legacy.nn.Sequential):
        torch_model = model
    else:
        raise TypeError(
            "Model must be file path to .t7 file or pytorch loaded model \
            with torch.legacy.nn.Sequential module as root"
        )

    torch_model.evaluate()

    if not isinstance(input_shape, tuple):
        raise TypeError("Input shape should be tuple.")

    output_shape = _infer_torch_output_shape(
        torch_model,
        input_shape
    )

    # create input/output features
    input_features = [(input_name, datatypes.Array(*input_shape))]
    output_features = [(output_name, datatypes.Array(*output_shape))]

    builder = NeuralNetworkBuilder(input_features, output_features, mode)

    # build model
    layer_name = _gen_layer_name(torch_model)
    _output_names = [output_name]
    if is_image_output:
        _output_names = [_DEPROCESS_LAYER_NAME]
    model_output_names = _layers._convert_layer(
        builder, layer_name, torch_model, [input_name], _output_names
    )

    # set preprocessing parameters
    if is_image_input:
        builder.set_pre_processing_parameters(
            image_input_names=[input_name],
            is_bgr=preprocessing_args.get('is_bgr', False),
            red_bias=preprocessing_args.get('red_bias', 0.0),
            green_bias=preprocessing_args.get('green_bias', 0.0),
            blue_bias=preprocessing_args.get('blue_bias', 0.0),
            gray_bias=preprocessing_args.get('gray_bias', 0.0),
            image_scale=preprocessing_args.get('image_scale', 1.0)
        )

    # set deprocessing parameters
    if is_image_output:
        if output_shape[0] == 1:
            is_grayscale = True
        elif output_shape[0] == 3:
            is_grayscale = False
        else:
            raise ValueError('Output must be RGB image or Grayscale')
        _set_deprocessing(
            is_grayscale,
            builder,
            deprocessing_args,
            model_output_names[0],
            output_name
        )

    if class_labels is not None:
        if type(class_labels) is str:
            labels = [l.strip() for l in open(class_labels).readlines()]
        elif type(class_labels) is list:
            labels = class_labels
        else:
            raise TypeError(
                "synset variable of unknown type. Type found: {}. \
                Expected either string or list of strings."
                .format(type(class_labels),))

        builder.set_class_labels(
            class_labels=labels,
            predicted_feature_name=predicted_feature_name
        )

    return MLModel(builder.spec)
