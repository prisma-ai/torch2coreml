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
        if len(array_shape) == 2:
            height, width = array_shape
            output.type.imageType.colorSpace = \
                ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        else:
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
