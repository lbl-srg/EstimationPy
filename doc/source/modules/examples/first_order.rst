First-Order system 
==================

The example demonstrates how to use **estimationpy** to run simulations,
pools of simulations in parallel, and solve a state estimation problem.
The examples investigated here uses the following LTI
system described by equations

.. math::

    \dot{x}(t) &= ax(t) + bu(t),\\
          y(t) &= cx(t) + du(t),\\
	  x(t_0)&= x_0

where :math:`x_0=2`, :math:`a=-1`, :math:`b=2.5`, :math:`c=3`, and :math:`d=0.1`.

The model can be written in Modelica as

.. literalinclude:: /../../estimationpy/modelica/FmuExamples/FirstOrder.mo
   :language: modelica
   :linenos:

and can be exported as a Functional Mockup Unit (FMU).
This model has been exported using Dymola (Linux version, both 32 and 64 bits)
and is located in the project folder.

Run a simulation
++++++++++++++++

The first example shows how to run a simple simulation
using the FMU model.

.. literalinclude:: /../../estimationpy/examples/first_order/run_model.py
   :language: python
   :linenos:
   :lines:  14-16, 28-30, 40-50

The model is instantiated in line 2, then at line 5 the path of the
FMU file is provided.
The input data is located in a CSV file called ``SimulationData_FirstOrder.csv``.
First the input of the model is selected by its name ``u``, and its reference
is the object ``input_u``.
After, the CSV file is associated to the input variable (line 10), and the
name of the column is indicated in line 11.

The instructions at line 14 and 17 respectively initialize the model, and run the
simulation. The figure below shows the result.

.. image:: ../../img/FirstOrder.png


Run multiple simulations
++++++++++++++++++++++++


State estimation
++++++++++++++++

