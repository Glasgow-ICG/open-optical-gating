# JT: Note that this seems to be required for pip installation,
# even though some of it duplicates information in pyproject.toml
# If there is a way to avoid requiring this duplication, I have not figured it out yet

from distutils.core import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

'''
    setuptools.setup(
                 name="open-optical-gating", # Replace with your own username
                 version="2.0.0b",
                 author="Alex Drysdale, Patrick Cameron, Jonathan Taylor and Chas Nelson",
                 author_email="",
                 description="Open-source prospective and adaptive optical gating for 3D fluorescence microscopy of beating hearts",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/Glasgow-ICG/open-optical-gating/",
                 packages=setuptools.find_packages(),
                 classifiers=[
                              "Programming Language :: Python :: 3",
                              "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                              "Operating System :: OS Independent",
                              ],
                 python_requires='>=3.7',
                 )
'''
setup(
      name="open-optical-gating", # Replace with your own username
      version="2.0.0b",
      author="Alex Drysdale, Patrick Cameron, Jonathan Taylor and Chas Nelson",
      description="Open-source prospective and adaptive optical gating for 3D fluorescence microscopy of beating hearts",
      long_description=long_description,
      url="https://github.com/Glasgow-ICG/open-optical-gating/",
      packages=['open_optical_gating', 'open_optical_gating.cli'],
      classifiers=[
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                   "Operating System :: OS Independent",
                   ],
      )
