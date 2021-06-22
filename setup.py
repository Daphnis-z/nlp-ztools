# -*- coding: utf-8 -*-
from distutils.core import setup

from setuptools import find_packages

LONGDOC = """
nlp tools
"""

setup(name='nlp-tools',
      version='1.0.0',
      description='nlp tools',
      long_description=LONGDOC,
      author='Daphnis',
      author_email='daphnisz@163.com',
      url='https://github.com/Daphnis-z',
      license="MIT",
      classifiers=[
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Natural Language :: Chinese (Traditional)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
          'Topic :: nlp tools'
      ],
      keywords='nlp',
      packages=find_packages('nlp'),
      package_dir={'': 'nlp'},
      package_data={'tool': ['*.*', '*/*']}
      )
