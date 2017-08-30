# Convert Torch7 models into Apple CoreML format.

This tool helps convert Torch7 models into [Apple CoreML](https://developer.apple.com/documentation/coreml) format which can then be run on Apple devices.

## Installation
```bash
pip install -U torch2coreml
```

In order to use this tool you need to have these installed:
* MacOS - High Sierra 10.13
* Xcode 9
* python 2.7

## Dependencies

* coremltools (0.5.0+)
* PyTorch

## How to use
Using this library you can implement converter for your own model types. An example of such a converter is located at "example/fast-neural-style/convert-fast-neural-style.py".
To implement converters you should use single function "convert" from torch2coreml:

```python
from torch2coreml import convert
```

This function is simple enough to be self-describing:

```python
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
            predicted_feature_name='classLabel')
```

### Parameters
__model__: Torch7 model (loaded with PyTorch) | str  
    A trained Torch7 model loaded in python using PyTorch or path to file
    with model (*.t7).

__input_shape__: tuple  
    Shape of the input tensor.

__mode__: str ('classifier', 'regressor' or None)  
    Mode of the converted coreml model:  
    'classifier', a NeuralNetworkClassifier spec will be constructed.  
    'regressor', a NeuralNetworkRegressor spec will be constructed.

__preprocessing_args__: dict  
    'is_bgr', 'red_bias', 'green_bias', 'blue_bias', 'gray_bias',
    'image_scale' keys with the same meaning as
    https://apple.github.io/coremltools/generated/coremltools.models.neural_network.html#coremltools.models.neural_network.NeuralNetworkBuilder.set_pre_processing_parameters

__deprocessing_args__: dict  
    Same as 'preprocessing_args' but for deprocessing.

__class_labels__: A string or list of strings.  
    As a string it represents the name of the file which contains
    the classification labels (one per line).
    As a list of strings it represents a list of categories that map
    the index of the output of a neural network to labels in a classifier.

__predicted_feature_name__: str  
    Name of the output feature for the class labels exposed in the Core ML
    model (applies to classifiers only). Defaults to 'classLabel'

### Returns
model: A coreml model.

## Currently supported
### Models
Only Torch7 "nn" module is supported now.

### Layers
List of Torch7 layers that can be converted into their CoreML equivalent:

1. Sequential
2. ConcatTable
3. SpatialConvolution
4. ELU
5. ReLU
6. SpatialBatchNormalization
7. Identity
8. CAddTable
9. SpatialFullConvolution
10. SpatialSoftMax
11. SpatialMaxPooling
12. SpatialAveragePooling
13. View
14. Linear
15. Tanh
16. MulConstant
17. SpatialZeroPadding
18. Narrow

## License

Copyright (c) 2017 Prisma Labs, Inc. All rights reserved.

Use of this source code is governed by the [MIT License](https://opensource.org/licenses/MIT) that can be found in the LICENSE.txt file.
