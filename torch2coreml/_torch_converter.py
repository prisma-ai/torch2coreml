import numpy as np
import torch

from torch.utils.serialization import load_lua

from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import MLModel, datatypes

import _layers

from _layers import _get_layer_converter_fn

from _utils import _gen_layer_name
from _utils import _convert_multiarray_output_to_image


_DEPROCESS_LAYER_NAME = 'deprocess_image'


def _forward_torch_random_input(torch_model, input_shapes, is_batch=False):
    input_tensors = []
    for shape in input_shapes:
        if is_batch:
            tensor = torch.rand(1, *shape).float()
        else:
            tensor = torch.rand(*shape).float()
        input_tensors.append(tensor)

    if len(input_tensors) == 1:
        result = torch_model.forward(input_tensors[0])
    else:
        result = torch_model.forward(input_tensors)

    if isinstance(result, list):
        # multi output
        output_shapes = []
        for tensor in result:
            shape = tensor.numpy().shape
            if is_batch:
                shape = shape[1:]
            output_shapes.append(shape)
        return output_shapes
    else:
        # single output
        output_shape = result.numpy().shape
        if is_batch:
            return [output_shape[1:]]
        else:
            return [output_shape]


def _infer_torch_output_shapes(torch_model, input_shapes):
    """
    Forward torch model to infer output shape
    """
    try:
        return _forward_torch_random_input(
            torch_model,
            input_shapes,
            is_batch=False
        )
    except:
        # try batch mode
        return _forward_torch_random_input(
            torch_model,
            input_shapes,
            is_batch=True
        )


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
        name=input_name,
        W=W,
        b=b,
        has_bias=True,
        shape_scale=W.shape,
        shape_bias=b.shape,
        input_name=input_name,
        output_name=output_name
    )


def convert(model,
            input_shapes,
            input_names=['input'],
            output_names=['output'],
            mode=None,
            image_input_names=[],
            preprocessing_args={},
            image_output_names=[],
            deprocessing_args={},
            class_labels=None,
            predicted_feature_name='classLabel',
            unknown_layer_converter_fn=None):
    """
    Convert Torch7 model to CoreML.

    Parameters
    ----------
    model: Torch7 model (loaded with PyTorch) | str
        A trained Torch7 model loaded in python using PyTorch or path to file
        with model (*.t7).

    input_shapes: list of tuples
        Shapes of the input tensors.

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

    unknown_layer_converter_fn: function with signature:
        (builder, name, layer, input_names, output_names)
            builder: object - instance of NeuralNetworkBuilder class
            name: str - generated layer name
            layer: object - pytorch object for corresponding layer
            input_names: list of strings
            output_names: list of strings
            Returns: list of strings for layer output names
        Callback function to handle unknown for torch2coreml layers


    Returns
    -------
    model: A coreml model.
    """
    _gen_layer_name.called = 0
    _get_layer_converter_fn.unknown_converter_fn = unknown_layer_converter_fn

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

    if not isinstance(input_shapes, list):
        raise TypeError("Input shapes should be a list of tuples.")

    for shape in input_shapes:
        if not isinstance(shape, tuple):
            raise TypeError("Input shape should be a tuple.")

    if len(input_names) != len(input_shapes):
        raise ValueError(
            "Input names count must be equal to input shapes count"
        )

    output_shapes = _infer_torch_output_shapes(
        torch_model,
        input_shapes
    )

    if len(output_shapes) != len(output_names):
        raise ValueError(
            "Model has {} outputs, but you set output_names for {}."
            .format(len(output_shapes), len(output_names))
        )

    # create input/output features
    input_features = []
    for i in range(len(input_names)):
        input_features.append(
            (input_names[i], datatypes.Array(*input_shapes[i]))
        )
    output_features = []
    for i in range(len(output_names)):
        output_features.append(
            (output_names[i], datatypes.Array(*output_shapes[i]))
        )

    builder = NeuralNetworkBuilder(input_features, output_features, mode)

    # build model
    layer_name = _gen_layer_name(torch_model)
    _output_names = output_names[:]
    if len(image_output_names) > 0:
        for i in range(len(_output_names)):
            if _output_names[i] in image_output_names:
                _output_names[i] = _gen_layer_name(_DEPROCESS_LAYER_NAME)

    model_output_names = _layers._convert_layer(
        builder, layer_name, torch_model, input_names, _output_names
    )

    # set preprocessing parameters
    if len(image_input_names) > 0:
        builder.set_pre_processing_parameters(
            image_input_names=image_input_names,
            is_bgr=preprocessing_args.get('is_bgr', False),
            red_bias=preprocessing_args.get('red_bias', 0.0),
            green_bias=preprocessing_args.get('green_bias', 0.0),
            blue_bias=preprocessing_args.get('blue_bias', 0.0),
            gray_bias=preprocessing_args.get('gray_bias', 0.0),
            image_scale=preprocessing_args.get('image_scale', 1.0)
        )

    # set deprocessing parameters
    if len(image_output_names) > 0:
        for i in range(len(output_names)):
            output_name = output_names[i]
            if output_name in image_output_names:
                output_shape = output_shapes[i]
                if len(output_shape) == 2 or output_shape[0] == 1:
                    is_grayscale = True
                elif output_shape[0] == 3:
                    is_grayscale = False
                else:
                    raise ValueError('Output must be RGB image or Grayscale')
                _set_deprocessing(
                    is_grayscale,
                    builder,
                    deprocessing_args,
                    model_output_names[i],
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
