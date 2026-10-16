from setuptools import find_packages
from distutils.core import setup

setup(
    name="legged_gym",
    version="1.0.0",
    author="Leon",
    author_email="renliang@joyin.ai",
    license="LGPL-3.0",
    packages=find_packages(),
    install_requires=[
        "isaacgym",
        "rsl-rl",
        "matplotlib",
        "numpy==1.21.1",
        "tensorboard",
    ]
)
