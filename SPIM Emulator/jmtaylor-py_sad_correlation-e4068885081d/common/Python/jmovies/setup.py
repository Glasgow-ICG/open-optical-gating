
from distutils.core import setup, Extension

#
# Global parameters
#
ARCH='i386'
ARCH2='x86_64'

BUILD_MODULES=[]



jmovies = Extension('jmoviesmodule',
	include_dirs = ['/usr/local/include',
					'/Library/Python/2.6/site-packages/numpy/core/include/'],
	library_dirs=['/usr/local/lib'],
	sources = ['jmovies.cpp', '../../JPythonArray.cpp', '../../JPythonCommon.cpp', '../../jAssert.cpp', '../../JMovieReader.cpp', ],
	extra_link_args=['-arch', ARCH -F],
	extra_compile_args=['-O3', '-arch', ARCH, '-include' 'prefix.h']
)
BUILD_MODULES.append(jmovies)


setup (ext_modules = BUILD_MODULES)




