import sys
import subprocess

from jinja2 import Environment


_INPUT_SHAPE = (3, 224, 224)


_SINGLE_LAYER_LUA_TEMPLATE = \
    """
    require 'torch'
    require 'nn'

    torch.setdefaulttensortype('torch.FloatTensor')

    local net = nn.Sequential()
    net:add({{ layer }})
    net:evaluate()

    {% if input_shapes|length == 1 %}
    local input = torch.rand({{ input_shapes[0] }})
    {% else %}
    local input = {}
    {% for shape in input_shapes %}
    table.insert(input, torch.rand({{ shape }}))
    {% endfor %}
    {% endif %}

    net:forward(input)

    torch.save('{{ model_path }}', net)
    """


def _generate_single_layer_torch_model(layer,
                                       input_shapes,
                                       model_path):
    input_shapes_str = [','.join(map(str, s)) for s in input_shapes]

    script = Environment().from_string(_SINGLE_LAYER_LUA_TEMPLATE).render(
        layer=layer,
        input_shapes=input_shapes_str,
        model_path=model_path
    )

    returncode = subprocess.call(['th', '-e', script],
                                 stdout=sys.stdout, stderr=sys.stderr)
    if returncode != 0:
        raise ValueError('Unsuccessful executing torch')
