from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='NYCtaxifarePredictor',
      version="1.0",
      description="Module for predicting NYC taxifare",
      url = 'https://github.com/Guli-Y/NYCtaxifarePredictor',
      author = 'Guli-Y',
      author_email = 'g.yimingjiang@gmail.com',
      install_requires=requirements,
      packages=find_packages(),
      test_suite = 'tests',
      include_package_data=True,
      scripts=['scripts/NYCtaxifarePredictor-run'],
      zip_safe=False)
