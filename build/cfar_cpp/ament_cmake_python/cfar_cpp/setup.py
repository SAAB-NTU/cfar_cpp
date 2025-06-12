from setuptools import find_packages
from setuptools import setup

setup(
    name='cfar_cpp',
    version='0.0.0',
    packages=find_packages(
        include=('cfar_cpp', 'cfar_cpp.*')),
)
