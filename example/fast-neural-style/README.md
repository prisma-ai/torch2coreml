# Convert fast-neural-style models into Apple CoreML format.

[Short tutorial](https://blog.prismalabs.ai/diy-prisma-app-with-coreml-6b4994cc99e1)

This is an example of converting original [fast-neural-style](https://github.com/jcjohnson/fast-neural-style) models into CoreML format.

# Usage

First of all you need to install torch2coreml:

```bash
pip install -U torch2coreml
```

Run __setup.sh__ script and wait until repo and models are downloaded, prepared and converted to CoreML. After that there should be a directory __coreml_models__ with style-transfer models which are ready to use in iOS app located in __ios__ directory. You need Xcode 9 and iOS 11 to test it.

If you want to test CoreML models from console, run:

```bash
python stylize-image.py -input input.jpg -output output.jpg -model coreml_models/starry_night.mlmodel
```

Replace input.jpg, output.jpg, coreml_models/starry_night.mlmodel with your values if you need it.
