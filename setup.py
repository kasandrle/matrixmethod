from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='matrixmethod_cython',
    ext_modules=cythonize("matrixmethod_cython.pyx"),
    include_dirs=[numpy.get_include()]
)
