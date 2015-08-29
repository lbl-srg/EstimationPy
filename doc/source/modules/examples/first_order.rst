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
simulation. The simulation command does not have arguments thus the time model
is simulated for the time period specified in the CSV file.
The figure below shows the result.

.. image:: ../../img/FirstOrder.png


Run multiple simulations
++++++++++++++++++++++++

The second example shows how to run a pool of models that use
all the same FMU model.

.. literalinclude:: /../../estimationpy/examples/first_order/run_pool.py
   :language: python
   :linenos:
   :lines:  16-18, 30-59
   :emphasize-lines: 19-20, 32-33 

In this case the only difference with respect tp the previosu case is that instead
of calling directly the :func:`estimationpy.fmu_utils.model.Model.simulate` method,
we define a :class:`estimationpy.fmu_utils.fmu_pool.FmuPool` object.
Then in lines 26-30 we create 10 different initial conditions for the
state vector, and in line 33 we run the simulation. The Figure below shows
the results of the 10 different simulations that are executed in parallel.
		     
.. image:: ../../img/FirstOrderPool.png

	   
State estimation
++++++++++++++++

The third example shows how to solve a state estimation problem with
estimationpy and an FMU model. Please note that in this example the model
used by the state estimation algorithm has been parametrized with values
for :math:`a`, :math:`b`, :math:`c`, and :math:`d` that are different from
the ones that were used to generate the measurements. This different represents
a case of model mismatch between the real system and its model.

In this case we have access to a set of measurements and we desire to estimate
the unknown state of the model. We have available a measurement of the input variable
:math:`u(t)` corrupted by noise :math:`u_m(t)`. Similarly, we don't know the exact
value of the output :math:`y(t)` but we have a measure of it :math:`y_m(t)`.
The state estimation algorithm, implemented using an Unscented Kalman Filter,
uses these measurements together with the model to estimate the probability
distribution of the state variable :math:`x(t)` and of the output :math:`y(t)`.
We indicates the estimated values with a hat, :math:`\hat{x}(t)` and
:math:`\hat{y}(t)`.

.. literalinclude:: /../../estimationpy/examples/first_order/run_ukf.py
   :language: python
   :linenos:
   :lines:  30-31, 36-57, 72-81

The code snippet shows how this problemis solved. As in the previous cases
we associate a column of the CSV file (in this case a CSV file with noisy
data) to the input ``u`` of the FMU model (lines 4-7).
However in this case we also need to specify which is the measured output, and we
which measured data is associated to it. Also, we can specify the covariance
of this data that can be seen as a proxy of the data quality.

After, in line 17, we specify that the model is used for estimating one state variable,
whose name is ``x``. In lines 20-24 we provide details for the state variable
to estimate

* initial value :math:`x_0 = 1.5`,
* covariance :math:`\sigma^2 = 0.5`,
* lower constraint :math:`x_{min} = 0`

As before we initialize the simulator (line 26), and then we instantiate an
object of type :class:`estimationpy.ukf.ufk_fmu.UfkFmu`.
At the end, at line 34, we start the filter and we specify its start and stop
time by providing two datetime objects. Please not that their time zone
is set to UTC.

The first plot shows the input data (blue dots) used by the UKF, and the measured
output data (green dots). The blue line and the green line are the true
values that were not made available to the UKF. The red line in the bottom plot
shows the value of the estimated output variable :math:`\hat{y}(t)`, and its
confidence interval.
		     
.. image:: ../../img/FirstOrder_InputOutput.png

The Figure below shows in green the unknown state variable :math:`x(t)` and
in red its estimation :math:`\hat{x}(t)`. The red area around the estimation indicates the
confidence interval of the estimation.
	   
.. image:: ../../img/FirstOrder_State.png
