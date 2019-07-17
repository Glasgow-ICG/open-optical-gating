from distutils.core import setup, Extension
import numpy	# So we can work out where the numpy headers live!
import platform
import os

# Work out if we should be building a 32 or 64 bit library
# Apparently this "can be a bit fragile" on OS X:
# http://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
# but I'll try it and see if it works out ok for now.
archInfo = platform.architecture()
if (archInfo[0] == '32bit'):
	ARCH = ['-arch', 'i386']
else:
	ARCH = ['-arch', 'x86_64']

# Determine if the -arch parameter is actually even available on this platform,
# by running a dummy gcc command that includes that option
# If it is not, then we will not include any arch-related options at all for gcc.
theString = 'gcc ' + ARCH[0] + ' ' + ARCH[1] + ' -E -dM - < /dev/null > /dev/null 2>&1'
result = os.system(theString)
if (result != 0):
	ARCH = []

BUILD_MODULES = []

j_py_sad_correlation = Extension('j_py_sad_correlation',
	include_dirs = ['/usr/local/include', numpy.get_include()],
	sources = ['j_py_sad_correlation.cpp', 'common/jPythonArray.cpp', 'common/jPythonCommon.cpp', 'common/PIVImageWindow.cpp', 'common/jAssert.cpp', 'common/DebugPrintf_Unix.cpp'],
	extra_link_args = ARCH,
	extra_compile_args = ['-O4', '-mssse3'] + ARCH
)
BUILD_MODULES.append(j_py_sad_correlation)

setup (ext_modules = BUILD_MODULES)