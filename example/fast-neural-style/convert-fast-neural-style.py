import argparse

from torch2coreml import convert


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

    coreml_model = convert(
        args.input,
        input_shape,
        input_name='inputImage',
        output_name='outputImage',
        is_image_input=True,
        preprocessing_args={
            'is_bgr': True,
            'red_bias': -123.68,
            'green_bias': -116.779,
            'blue_bias': -103.939
        },
        is_image_output=True,
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
