#pip setup file 

from setuptools import setup, find_packages

setup(
    name='semseg',
    version='0.1',
    packages=find_packages(where='.'),
)