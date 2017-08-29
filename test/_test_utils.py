import sys
import subprocess


_INPUT_SHAPE = (3, 224, 224)


_SINGLE_LAYER_LUA_TEMPLATE = \
    """
    require 'torch'
    require 'nn'

    torch.setdefaulttensortype('torch.FloatTensor')

    local net = nn.Sequential()
    net:add({layer})
    net:evaluate()

    local input = torch.rand({input_shape})
    net:forward(input)

    torch.save('{model_path}', net)
    """


def _generate_single_layer_torch_model(layer,
                                       input_shape,
                                       model_path):
    input_shape_str = ','.join(map(str, input_shape))

    script = _SINGLE_LAYER_LUA_TEMPLATE.format(
        layer=layer,
        input_shape=input_shape_str,
        model_path=model_path
    )

    returncode = subprocess.call(['th', '-e', script],
                                 stdout=sys.stdout, stderr=sys.stderr)
    if returncode != 0:
        raise ValueError('Unsuccessful executing torch')
