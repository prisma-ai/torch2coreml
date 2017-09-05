import argparse

from torch.legacy.nn import Module, SpatialFullConvolution
from torch.autograd import Variable
from torch.nn import InstanceNorm3d
from torch.utils.serialization import load_lua
from torch.utils.serialization.read_lua_file import TorchObject

from torch2coreml import convert


class InstanceNormalization(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(Module, self).__init__()
        self._instance_norm = InstanceNorm3d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=True
        )

    @property
    def running_mean(self):
        return self._instance_norm.running_mean

    @running_mean.setter
    def running_mean(self, value):
        self._instance_norm.running_mean = value

    @property
    def running_var(self):
        return self._instance_norm.running_var

    @running_var.setter
    def running_var(self, value):
        self._instance_norm.running_var = value

    @property
    def weight(self):
        return self._instance_norm.weight.data

    @weight.setter
    def weight(self, value):
        self._instance_norm.weight.data = value

    @property
    def bias(self):
        return self._instance_norm.bias.data

    @bias.setter
    def bias(self, value):
        self._instance_norm.bias.data = value

    def updateOutput(self, input):
        return self._instance_norm.forward(Variable(input, volatile=True)).data


def replace_module(module, check_fn, create_fn):
    if not hasattr(module, 'modules'):
        return
    if module.modules is None:
        return
    for i in range(len(module.modules)):
        m = module.modules[i]
        if check_fn(m):
            module.modules[i] = create_fn(m)
        replace_module(m, check_fn, create_fn)


def create_instance_norm(m):
    num_features = m.nOutput
    eps = m.eps
    momentum = m.momentum

    layer = InstanceNormalization(
        num_features,
        eps=eps,
        momentum=momentum,
        affine=True
    )

    layer.running_mean = m.running_mean
    layer.running_var = m.running_var
    layer.weight = m.weight
    layer.bias = m.bias

    return layer


def fix_full_conv(m):
    m.finput = None
    m.fgradInput = None
    return m


def load_torch_model(path):
    model = load_lua(path, unknown_classes=True)
    replace_module(
        model,
        lambda m: isinstance(m, TorchObject) and
        m.torch_typename() == 'nn.InstanceNormalization',
        create_instance_norm
    )
    replace_module(
        model,
        lambda m: isinstance(m, SpatialFullConvolution),
        fix_full_conv
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Convert fast-neural-style model to CoreML'
    )

    parser.add_argument('-input', required=True, help='Path to Torch7 model')
    parser.add_argument('-output', required=True, help='CoreML output path')
    parser.add_argument(
        '-size',
        default=720, type=int, help='Image width/height'
    )

    args = parser.parse_args()

    input_shape = (3, args.size, args.size)

    model = load_torch_model(args.input)

    coreml_model = convert(
        model,
        [input_shape],
        input_names=['inputImage'],
        output_names=['outputImage'],
        image_input_names=['inputImage'],
        preprocessing_args={
            'is_bgr': True,
            'red_bias': -123.68,
            'green_bias': -116.779,
            'blue_bias': -103.939
        },
        image_output_names=['outputImage'],
        deprocessing_args={
            'is_bgr': True,
            'red_bias': 123.68,
            'green_bias': 116.779,
            'blue_bias': 103.939
        }
    )

    coreml_model.author = 'Justin Johnson'
    coreml_model.license = 'Free for personal or research use'
    coreml_model.short_description = 'Feedforward style transfer \
https://github.com/jcjohnson/fast-neural-style'
    coreml_model.input_description['inputImage'] = 'Image to stylize'
    coreml_model.output_description['outputImage'] = 'Stylized image'

    coreml_model.save(args.output)


if __name__ == "__main__":
    main()
