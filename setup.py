# Copyright 2023 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from shutil import rmtree
import os

# remove the old dist folder
rmtree('dist', ignore_errors=True)
# remove the old documentation folder
rmtree('docs', ignore_errors=True)

os.system("pdoc --html src/decision_boundary_mapper -o docs")

setup(
    name='decision-boundary-mapper',
    version='0.3.4',
    license='MIT',
    author="Cristian Grosu",
    author_email='c.grosu@students.uu.nl',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/cristi2019255/MasterThesis2023',
    keywords='Decision Boundary Mapper',
    description='A tool for visualizing the decision boundary of a machine learning model.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
          'dask',
          'keras',
          'matplotlib',
          'numba',
          'numpy',
          'Pillow',
          'PySimpleGUI',
          'pydot',
          'scikit_learn',
          'scipy',
          'tensorflow',
          'termcolor',
          'umap',
          'umap-learn',
      ],
)