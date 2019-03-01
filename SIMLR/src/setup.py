from distutils.core import setup, Extension
import numpy

projsplx_c = Extension('projsplx_c', sources=['projsplx_c.c'], 
	include_dirs=[numpy.get_include()])
#Kbeta = Extension('Kbeta2', sources=['Kbeta2.c'],
#	include_dirs=[numpy.get_include()])

setup(ext_modules=[projsplx_c])
#setup(ext_modules=[Kbeta])