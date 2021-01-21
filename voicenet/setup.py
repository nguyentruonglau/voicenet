from setuptools import setup
import setuptools

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='voicenet',
      version='1.1.0',
      description='Convolutional neural network architecture found by nondominated sorting genetic algorithm ii with code: 0-10 - 1-01-001 - 0-00',
      long_description=readme(),
      classifiers=[
       "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
      ],
      keywords=' nondominated sorting genetic algorithm ii, neural architecture search, convolutional neural network',
      url='https://github.com/nguyentruonglau/VoiceNet',
      author='Nguyen Truong Lau',
      author_email='ttruongllau@gmail.com',
      license='UIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      zip_safe=False)