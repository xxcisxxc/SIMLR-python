from distutils.core import setup, Extension
import numpy

#projsplx_c = Extension('projsplx_c', sources=['projsplx_c.c'], 
#	include_dirs=[numpy.get_include()])
#top_eig = Extension('top_eig', sources=['top_eig.cpp'],
#	include_dirs=[numpy.get_include(), 'include'], language='c++')
compute_wtsne_obj_grad_repulsive_barneshut = Extension('compute_wtsne_obj_grad_repulsive_barneshut', sources=['compute_wtsne_obj_grad_repulsive_barneshut.cpp'],
	include_dirs=[numpy.get_include()])

#setup(ext_modules=[projsplx_c])
#setup(ext_modules=[top_eig])
setup(ext_modules=[compute_wtsne_obj_grad_repulsive_barneshut])
