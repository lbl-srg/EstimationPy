Applications
============

This pages contains some information about the FMI standard and how it can
be used in conjunction with **EstimationPy** to develop applications.

FMI
+++

The FMI standard is a tool-independent standard for the exchange of models and simulation programs.
The development of the FMI was initiated by Daimler within the project
`MODELISAR <https://itea3.org/project/modelisar.html>`_ ,
a research project funded by the European Union that
involved 29 partners from industry, simulation tool vendors, and academia. The goal of FMI was to
support the exchange of simulation models between manufacturers and suppliers in the automotive
sector. Manufacturers create cars that are made of different subsystems (such as wheels, engines,
and control systems) that are provided by different suppliers and then integrated into an automobile.
To test the performance of the overall system, the manufacturers needed to couple various simulation
models provided by the suppliers of each component. This was difficult and expensive, and it allowed
limited analysis. Today more than 40 simulation programs support the FMI standard.
The standard is governed by a group of companies, institutes, and universities, and is organized
through the Modelica Association.
Under the umbrella of the International Energy Agency (IEA), Energy in Buildings and Communities
Programme (EBC), a five-year project called Annex 60 started in 2012.
Annex 60 http://www.iea-annex60.org is an international
project, led by Lawrence Berkeley National Laboratory (LBNL) and by RWTH Aachen, Germany, titled
“New generation computational tools for building and community energy systems based on the
Modelica and Functional Mockup Interface standards.” In this project, 37 institutions from 16
countries are working together to coordinate their efforts and demonstrate that technologies
based on the FMI standard and the Modelica open modeling language can be successfully applied
to the design and operation of buildings and of community energy systems.
The use of the FMI standard in the context of control applications for buildings has been
demonstrated by Nouidui et al., [Noudui2014]_. They presented how to import FMI models into the
NiagaraAX framework®, a platform that allows integration of different building automation
systems. Also, the Building Control Virtual Test Bed [Wetter2011]_ allows the import of
models compliant with the FMI standard. This allows FMI models to couple with
BACnet-compatible building automation systems and with web services, either as a web
server or as a client.

Model-based fault detection
+++++++++++++++++++++++++++

This section presents the workflow for building Fault Detection and Diagnostic algorithms based on
EstimationPy and the FMI standard. We emphasize that the workflow uses models that were developed
during design, and thereby leverages the knowledge and experiences accrued while designing the system.
By reusing models from the design, users reduce the time and expertise (and therefore, costs) required
to set up, develop, and deploy an online FDD system.

.. figure:: img/FddWithFMI.png

The above Figure shows the workflow. During the design phase, indicated by the green box, designers
use their simulation program to design the building and its HVAC systems. The simulation program provides
different libraries of models that users can use to design the systems. An example could be using the
Modelica Buildings Library [Wetter2014]_ within a simulation program like Dymola [#1]_.
In the workflow, this phase does not require any additional effort if simulation models are used to
predict the energy performances of the building and its HVAC systems.
At the end of the design phase, if the results satisfy the design intent, the building is constructed,
its HVAC system is commissioned, and the building is occupied. This is indicated in the brown box in
Figure. An energy information system (EIS) records the performance of the building, and it becomes available
to building and to energy managers. The availability of an EIS or any other software that collects and
provides data from the building is an essential requirement for developing any FDD system.
Some FDD systems use measurements collected by the EIS system to check whether the building and its HVAC is
working as expected. The benefits are twofold: reducing the impact of the faults on the overall energy consumption
of the building, and avoiding serious damage to the equipment. During operation the same models utilized
in the design phase can be reused by FDD algorithm, the grey circle in Figure. This connection between the
FDD algorithm and the simulation program is made possible by the FMI standard interface. Once the model is
exported, the FDD algorithm uses it together with the data acquired by sensors and instrumentations
located in the building to identify faults that increase building energy consumption.

Footnotes
+++++++++
		
.. [#1]  http://www.3ds.com/products-services/catia/capabilities/systems-engineering/modelica-systems-simulation/dymola


References
++++++++++

.. [Noudui2014] Nouidui, T. S., Wetter, M. 2014.
		"Tool coupling for the design and operation of building energy and control systems based on the
		Functional Mockup Interface standard"
		10th International Modelica Conference, Lund, Sweden. March.

.. [Wetter2011] Wetter M. 2011.
		"Co-simulation of building energy and control systems with the Building
		Controls Virtual Test Bed."
		Journal of Building Performance Simulation 4(3):185–203.

.. [Wetter2014] Wetter, M., Zuo, W., Nouidui, T. S., Pang, X. 2014.
		"Modelica buildings library"
		Journal of Building Performance Simulation 7(4):253–270
