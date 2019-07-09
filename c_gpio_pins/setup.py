from distutils.core import setup, Extension
module1 = Extension('fastpins',
		include_dirs = ['/usr/local/'],
		libraries = ['wiringPi'],
		library_dirs = ['/usr/lib'],
		sources=['fastpinsmodule.c'])

setup (name='FastPins',
	version = '1.0',
	description= 'Very responsive (fast) pin triggering and reading for python.',
	author = 'Alex B Drysdale',
	ext_modules = [module1])
