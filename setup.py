import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
 
setup(
    name = 'estimationpy',
    version = '0.1.0',
    packages = ['estimationpy', 
                'estimationpy.tests',
                'estimationpy.ukf',
                'estimationpy.fmu_utils',
                'estimationpy.examples'],
    include_package_data = True,
    license = 'BSD License',
    description = 'A python package for state and parameter estimation \
compliant with the Functional Mockup Interface standard',
    long_description = README,
    url = 'http://www.estimationpy.lbl.gov/',
    author = 'Marco Bonvini',
    author_email = 'MBonvini@lbl.gov',
    classifiers =[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
    ]
)
