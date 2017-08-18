import sys
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_args = ['-Wall', '-O3']

ext_mod = Extension('py_grba_int',
                       sources=['py_grba_int.pyx', 'grba_int.cpp', './phi_int/phi_int.cpp', './r0_int/r0_int.cpp'],
                       libraries=['cminpack'],
                       extra_compile_args=compile_args,
                       language="c++")

setup(
    name='py_grba_int',
    ext_modules = cythonize(ext_mod),
    include_dirs = [numpy.get_include()]
)
