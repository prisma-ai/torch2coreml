from _utils import _gen_layer_name, _torch_typename
from _utils import _add_pooling_with_padding_types


def _convert_sequential(builder, name, layer, input_names, output_names):
    layers = layer.modules
    n = len(layers)

    inputs = input_names
    for i in range(n):
        l = layers[i]

        l_outputs = None
        l_name = _gen_layer_name(l)
        if i != (n - 1):
            l_outputs = [l_name]
        else:
            l_outputs = output_names

        l_outputs = _convert_layer(builder, l_name, l, inputs, l_outputs)
        inputs = l_outputs

    return output_names


def _convert_convolution(builder, name, layer, input_names, output_names):
    input_name = input_names[0]
    output_name = output_names[0]

    k_h, k_w = layer.kH, layer.kW

    pad_h, pad_w = layer.padH, layer.padW
    weight = layer.weight.numpy().transpose((2, 3, 1, 0))

    bias = None
    if layer.bias is not None:
        bias = layer.bias.numpy()

    if pad_h > 0 or pad_w > 0:
        padding_name = _gen_layer_name('padding')
        builder.add_padding(
            name=padding_name,
            left=pad_w,
            right=pad_w,
            top=pad_h,
            bottom=pad_h,
            value=0.0,
            input_name=input_name,
            output_name=padding_name
        )
        input_name = padding_name

    builder.add_convolution(
        name=name,
        kernel_channels=layer.nInputPlane,
        output_channels=layer.nOutputPlane,
        height=k_h,
        width=k_w,
        stride_height=layer.dH,
        stride_width=layer.dW,
        border_mode='valid',
        groups=1,
        W=weight,
        b=bias,
        has_bias=bias is not None,
        is_deconv=False,
        output_shape=None,
        input_name=input_name,
        output_name=output_name,
        dilation_factors=[1, 1]
    )

    return output_names


def _convert_full_convolution(builder, name, layer, input_names, output_names):
    input_name = input_names[0]
    output_name = output_names[0]

    k_h, k_w = layer.kH, layer.kW

    pad_h, pad_w = layer.padH, layer.padW

    weight = layer.weight.numpy().transpose((2, 3, 0, 1))
    bias = None
    if layer.bias is not None:
        bias = layer.bias.numpy()

    add_crop = False
    output_ = layer.output.numpy()
    output_shape = (
        output_.shape[-2] + 2 * pad_h,
        output_.shape[-1] + 2 * pad_w
    )
    if pad_h > 0 or pad_w > 0:
        crop_padding_name = _gen_layer_name('padding')
        output_name = name + '_output'
        add_crop = True

    builder.add_convolution(
        name=name,
        kernel_channels=layer.nInputPlane,
        output_channels=layer.nOutputPlane,
        height=k_h,
        width=k_w,
        stride_height=layer.dH,
        stride_width=layer.dW,
        border_mode='valid',
        groups=1,
        W=weight,
        b=bias,
        has_bias=bias is not None,
        is_deconv=True,
        output_shape=output_shape,
        input_name=input_name,
        output_name=output_name,
        dilation_factors=[1, 1]
    )

    if add_crop:
        builder.add_crop(
            name=crop_padding_name,
            left=pad_w,
            right=pad_w,
            top=pad_h,
            bottom=pad_h,
            offset=0,
            input_names=[output_name],
            output_name=output_names[0]
        )

    return output_names


def _convert_elu(builder, name, layer, input_names, output_names):
    builder.add_activation(
        name=name,
        non_linearity='ELU',
        input_name=input_names[0],
        output_name=output_names[0],
        params=layer.alpha
    )

    return output_names


def _convert_relu(builder, name, layer, input_names, output_names):
    builder.add_activation(
        name=name,
        non_linearity='RELU',
        input_name=input_names[0],
        output_name=output_names[0]
    )

    return output_names


def _convert_concat_table(builder, name, layer, input_names, output_names):
    layers = layer.modules
    result_outputs = []
    for l in layers:
        l_name = _gen_layer_name(l)
        l_outputs = _convert_layer(builder, l_name, l, input_names, [l_name])
        result_outputs += l_outputs
    return result_outputs


def _convert_batch_norm(builder, name, layer, input_names, output_names):
    epsilon = layer.eps
    mean = layer.running_mean.numpy()
    variance = layer.running_var.numpy()
    weight = layer.weight.numpy()

    bias = None
    if layer.bias is not None:
        bias = layer.bias.numpy()

    builder.add_batchnorm(
        name=name,
        channels=weight.shape[0],
        gamma=weight,
        beta=bias,
        mean=mean,
        variance=variance,
        input_name=input_names[0],
        output_name=output_names[0],
        epsilon=epsilon
    )

    return output_names


def _convert_cadd_table(builder, name, layer, input_names, output_names):
    assert len(input_names) > 1
    assert len(output_names) == 1
    builder.add_elementwise(
        name=name,
        input_names=input_names,
        output_name=output_names[0],
        mode='ADD'
    )

    return output_names


def _convert_identity(builder, name, layer, input_names, output_names):
    return input_names


def _convert_soft_max(builder, name, layer, input_names, output_names):
    builder.add_softmax(
        name=name,
        input_name=input_names[0],
        output_name=output_names[0]
    )
    return output_names


def _convert_pooling(builder, name, layer, input_names, output_names):
    typename = _torch_typename(layer)

    exclude_pad_area = True

    if typename == 'SpatialMaxPooling':
        layer_type = 'MAX'
    elif typename == 'SpatialAveragePooling':
        layer_type = 'AVERAGE'
        exclude_pad_area = not layer.count_include_pad
    else:
        raise TypeError("Unknown type '{}'".format(typename,))

    k_h, k_w = layer.kH, layer.kW
    pad_h, pad_w = layer.padH, layer.padW
    d_h, d_w = layer.dH, layer.dW

    _add_pooling_with_padding_types(
        builder=builder,
        name=name,
        height=k_h,
        width=k_w,
        stride_height=d_h,
        stride_width=d_w,
        layer_type=layer_type,
        padding_type='VALID',
        input_name=input_names[0],
        output_name=output_names[0],
        exclude_pad_area=exclude_pad_area,
        padding_top=pad_h,
        padding_bottom=pad_h,
        padding_left=pad_w,
        padding_right=pad_w
    )

    return output_names


def _convert_linear(builder, name, layer, input_names, output_names):
    weight = layer.weight.numpy()
    bias = layer.bias.numpy()

    has_bias = bias is not None

    output_channels, input_channels = weight.shape

    builder.add_inner_product(
        name=name,
        W=weight,
        b=bias,
        input_channels=input_channels,
        output_channels=output_channels,
        has_bias=has_bias,
        input_name=input_names[0],
        output_name=output_names[0]
    )

    return output_names


def _convert_view(builder, name, layer, input_names, output_names):
    shape = tuple(layer.size)
    if len(shape) == 1 or (len(shape) == 2 and shape[0] == 1):
        builder.add_flatten(
            name=name,
            mode=0,
            input_name=input_names[0],
            output_name=output_names[0]
        )
    else:
        builder.add_reshape(
            name=name,
            input_name=input_names[0],
            output_name=output_names[0],
            target_shape=shape,
            mode=0
        )

    return output_names


def _convert_tanh(builder, name, layer, input_names, output_names):
    builder.add_activation(
        name=name,
        non_linearity='TANH',
        input_name=input_names[0],
        output_name=output_names[0],
        params=None
    )
    return output_names


def _convert_mul_constant(builder, name, layer, input_names, output_names):
    scalar = float(layer.constant_scalar)

    nn_spec = builder.nn_spec
    scale_layer = nn_spec.layers.add()
    scale_layer.name = name
    scale_layer.input.append(input_names[0])
    scale_layer.output.append(output_names[0])
    scale_layer.activation.linear.alpha = scalar

    return output_names


def _convert_zero_padding(builder, name, layer, input_names, output_names):
    pad_l = int(layer.pad_l)
    pad_r = int(layer.pad_r)
    pad_t = int(layer.pad_t)
    pad_b = int(layer.pad_b)

    if pad_l < 0 and pad_r < 0 and pad_t < 0 and pad_b < 0:
        # crop mode
        builder.add_crop(
            name=name,
            left=-pad_l,
            right=-pad_r,
            top=-pad_t,
            bottom=-pad_b,
            offset=None,
            input_names=input_names,
            output_name=output_names[0]
        )
    else:
        builder.add_padding(
            name=name,
            left=pad_l,
            right=pad_r,
            top=pad_t,
            bottom=pad_b,
            value=0.0,
            input_name=input_names[0],
            output_name=output_names[0]
        )

    return output_names


def _convert_narrow(builder, name, layer, input_names, output_names):
    dimension = layer.dimension
    if len(layer.output.numpy().shape) == 4:
        # as torch layer works with 4d tensor we should decrement dimension
        dimension -= 1

    if dimension == 0:
        axis = 'channel'
    elif dimension == 1:
        axis = 'height'
    elif dimension == 2:
        axis = 'width'
    else:
        raise ValueError('Only 3d tensors are supported')

    index = layer.index
    length = layer.length

    builder.add_slice(
        name=name,
        axis=axis,
        start_index=index,
        end_index=index + length,
        stride=1,
        input_name=input_names[0],
        output_name=output_names[0]
    )

    return output_names


_TORCH_LAYER_REGISTRY = {
    'Sequential': _convert_sequential,
    'SpatialConvolution': _convert_convolution,
    'ELU': _convert_elu,
    'ConcatTable': _convert_concat_table,
    'SpatialBatchNormalization': _convert_batch_norm,
    'Identity': _convert_identity,
    'CAddTable': _convert_cadd_table,
    'SpatialFullConvolution': _convert_full_convolution,
    'SpatialSoftMax': _convert_soft_max,
    'SoftMax': _convert_soft_max,
    'ReLU': _convert_relu,
    'SpatialMaxPooling': _convert_pooling,
    'SpatialAveragePooling': _convert_pooling,
    'View': _convert_view,
    'Linear': _convert_linear,
    'Tanh': _convert_tanh,
    'MulConstant': _convert_mul_constant,
    'SpatialZeroPadding': _convert_zero_padding,
    'Narrow': _convert_narrow
}


def _get_layer_converter_fn(layer):
    """
    Get the right converter function for Torch layer name
    """
    name = _torch_typename(layer)
    if name in _TORCH_LAYER_REGISTRY:
        return _TORCH_LAYER_REGISTRY[name]
    else:
        raise TypeError(
            "Torch layer of type {} is not supported.".format(name,)
        )


def _convert_layer(builder, name, layer, input_names, output_names):
    converter_fn = _get_layer_converter_fn(layer)
    return converter_fn(builder, name, layer, input_names, output_names)
