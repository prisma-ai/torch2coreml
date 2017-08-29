from coremltools.proto import NeuralNetwork_pb2 as _NeuralNetwork_pb2
from coremltools.proto import FeatureTypes_pb2 as ft


def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        return fn(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper


@counted
def _gen_layer_name(layer):
    _last_index = _gen_layer_name.called
    if isinstance(layer, str):
        return "{}_{}".format(layer, _last_index)
    typename = _torch_typename(layer)
    return "{}_{}".format(typename, _last_index)


def _torch_typename(obj):
    return type(obj).__name__


def _convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError(
                "{} is not a multiarray type".format(output.name,)
            )
        array_shape = tuple(output.type.multiArrayType.shape)
        channels, height, width = array_shape

        if channels == 1:
            output.type.imageType.colorSpace = \
                ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        elif channels == 3:
            if is_bgr:
                output.type.imageType.colorSpace = \
                    ft.ImageFeatureType.ColorSpace.Value('BGR')
            else:
                output.type.imageType.colorSpace = \
                    ft.ImageFeatureType.ColorSpace.Value('RGB')
        else:
            raise ValueError(
                "Channel Value {} not supported for image inputs"
                .format(channels,)
            )

        output.type.imageType.width = width
        output.type.imageType.height = height


def _add_pooling_with_padding_types(
        builder, name, height, width, stride_height, stride_width,
        layer_type, padding_type, input_name, output_name,
        padding_top=0, padding_bottom=0, padding_left=0, padding_right=0,
        same_padding_asymmetry_mode='BOTTOM_RIGHT_HEAVY',
        exclude_pad_area=True, is_global=False):
    """
    This is our own implementation of add_pooling since current CoreML's
    version (0.5.0) of builder doesn't provide support for padding types apart
    from valid. This support will be added in the next release of coremltools.
    When that happens, this can be removed.
    https://github.com/apache/incubator-mxnet/blob/master/tools/coreml/converter/_add_pooling.py
    """
    nn_spec = builder.nn_spec

    # Add a new layer
    spec_layer = nn_spec.layers.add()
    spec_layer.name = name
    spec_layer.input.append(input_name)
    spec_layer.output.append(output_name)
    layer_params = spec_layer.pooling

    # Set the parameters
    layer_params.type = \
        _NeuralNetwork_pb2.PoolingLayerParams.PoolingType.Value(layer_type)

    if padding_type == 'VALID':
        height_border = layer_params.valid.paddingAmounts.borderAmounts.add()
        height_border.startEdgeSize = padding_top
        height_border.endEdgeSize = padding_bottom
        width_border = layer_params.valid.paddingAmounts.borderAmounts.add()
        width_border.startEdgeSize = padding_left
        width_border.endEdgeSize = padding_right
    elif padding_type == 'SAME':

        if not (same_padding_asymmetry_mode == 'BOTTOM_RIGHT_HEAVY' or
                same_padding_asymmetry_mode == 'TOP_LEFT_HEAVY'):
            raise ValueError(
                "Invalid value {} of same_padding_asymmetry_mode parameter"
                .format(same_padding_asymmetry_mode,))
        _spm = _NeuralNetwork_pb2.SamePadding.SamePaddingMode
        layer_params.same.asymmetryMode = \
            _spm.Value(same_padding_asymmetry_mode)
    elif padding_type == 'INCLUDE_LAST_PIXEL':
        if padding_top != padding_bottom or padding_left != padding_right:
            raise ValueError(
                "Only symmetric padding is supported \
                with the INCLUDE_LAST_PIXEL padding type"
            )
        layer_params.includeLastPixel.paddingAmounts.append(padding_top)
        layer_params.includeLastPixel.paddingAmounts.append(padding_left)

    layer_params.kernelSize.append(height)
    layer_params.kernelSize.append(width)
    layer_params.stride.append(stride_height)
    layer_params.stride.append(stride_width)
    layer_params.avgPoolExcludePadding = exclude_pad_area
    layer_params.globalPooling = is_global
