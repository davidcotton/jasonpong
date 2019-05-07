import os
from setuptools import setup

NAME = 'jason_pong'


def read_file(path):
    with open(os.path.join(os.path.dirname(__file__), path)) as handle:
        return handle.read()


def get_version(path):
    version_line = [line for line in open(path) if line.startswith('__version__')][0]
    return version_line.split('__version__ = ')[-1][1:][:-2]


setup(name=NAME,
      version=get_version(os.path.join(NAME, '__init__.py')),
      description='Jason Pong',
      long_description=open('README.md').read(),
      install_requires=['gym'],
      )
