from setuptools import setup, find_packages
from codecs import open
from os import path


VERSION = '0.0.4'

here = path.abspath(path.dirname(__file__))

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='torch2coreml',
    version=VERSION,
    packages=find_packages(exclude=['contrib', 'docs', 'test', 'example']),
    description="Convert Torch7 models into Apple CoreML format.",
    long_description=long_description,
    url='https://github.com/prisma-ai/torch2coreml/',
    author='Oleg Poyaganov',
    author_email='oleg@prisma-ai.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    keywords='coreml machinelearning ml coremltools torch converter neural',
    python_requires='==2.7.*',
    install_requires=[
        'coremltools>=0.5.0',
        'torch'
    ]
)
