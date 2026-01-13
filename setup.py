from setuptools import setup, find_packages
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='online_gp',
    version='0.0.17',
    description=('Repo for Streaming GPs'),
    long_description=long_description,
    author='',
    author_email='',
    url='',
    license='MPL-2.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    classifiers=[
        'Development Status :: 0',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7'],
)