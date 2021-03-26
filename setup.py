import sys
import subprocess
import re
import setuptools

from codecs import open
from os.path import (abspath, dirname, join)

import builtins

builtins.__ISABELAFUNCTIONS_SETUP__ = True

exec(open('IsabelaFunctions/version.py').read())

setuptools.setup(
    name = "IsabelaFunctions",
    version=__version__,

    author = "Isabela de Oliveira",
    author_email = "deoliveira.isabela@outlook.com",

    keywords = "Functions",
    description = "My functions",
    
    packages = setuptools.find_packages(),

    classifiers = [
        "Development Status :: Beta ",

        "Natural Language :: English",
        "Programming Language :: Python : : 3",
        
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering ::Magnetic Fields",
        "license :: OSI Approved :: MIT License",
    ],

    install_requires = [
        "scipy",
        "numpy",
        "matplotlib",
        "pyshtools",
    ],

)
