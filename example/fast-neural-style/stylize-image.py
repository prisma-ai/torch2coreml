import argparse

from PIL import Image
from coremltools.models import MLModel


def main():
    parser = argparse.ArgumentParser(
        description='Stylize image using CoreML'
    )

    parser.add_argument('-input', required=True, help='Path to input image')
    parser.add_argument('-output', required=True, help='Output path')
    parser.add_argument('-model', required=True, help='CoreML model path')

    args = parser.parse_args()

    image = Image.open(args.input)

    net = MLModel(args.model)
    stylized_image = net.predict({'inputImage': image})['outputImage']
    stylized_image = stylized_image.convert('RGB')

    stylized_image.save(args.output)


if __name__ == "__main__":
    main()
