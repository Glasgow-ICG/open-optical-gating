'''
    There appears to be a bug in setuptools (Nov 2020) which
    prevents us from using this supposedly more modern approach:
        https://github.com/pypa/setuptools/issues/2353
        https://stackoverflow.com/questions/63683262/modulenotfounderror-no-module-named-setuptools-distutils
    Workaround is to delete pyproject.toml (but we need it…!?),
    or I could just revert to not using setuptools, which is what I have done for now
'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

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
