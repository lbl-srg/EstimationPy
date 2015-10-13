.. EstimationPy documentation master file, created by
   sphinx-quickstart on Tue Sep 17 11:46:08 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: img/EstimationPy.png

EstimationPy is a Python package for state and parameter estimation
of dynamic systems that conform to the Functional Mockup Interface
standard `FMI <http://www.fmi-standard.org>`_.

EstimationPy relies on Python numerical packages such as 
`NumPy <http://www.numpy.org/>`_ and 
`SciPy <http://www.scipy.org/>`_ for
performing computations, and it is compatible with
`Pandas <http://www.pandas.pydata.org/>`_ DataFrames and DataSeries.

EstimationPy strongly relies on `PyFMI <http://www.pyfmi.org/>`_ and
`Assimulo <http://www.assimulo.org/>`_ for running the simulations
of the models.

Assumptions
+++++++++++

The package assumes that all the data series and the data imported
from CSV files use UTC as time zone. Please make sure that when you
associate a **pandas.Series** to an input or output it uses
UTC as timezone. The examples and the unit tests show how this can
be done.

Models
++++++

EstimationPy contains a set of Modelica models that have been used
by the examples and the unit tests. These models have
already been exported as FMUs.

Source code
+++++++++++

The source code of EstimationPy and this documentation is available
here https://github.com/lbl-srg/EstimationPy .
This documentation is generated automatically and hosted
on the branch **gh-pages** of the repository.

See the section :ref:`Frequently-Asked-Questions` for instructions on how to build
the docs.

**Contents**

.. toctree::
  :maxdepth: 2
	     
  installation
  modules/fmu_utils
  modules/ukf
  modules/examples
  applications
  faq
  publications
  contribute
  contributors
  license 
  legal
  
.. automodule:: estimationpy.ukf
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

