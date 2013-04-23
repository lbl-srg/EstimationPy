# EETD - ESTCP NDW EIS
## Environmental Energy Technology Division
## ESTCP project for Fault Detection and Diagnosis
## Naval District Washington D.C.
## Energy Information System

This project contains different resources for the project, mainly python and Modelica code. The resources are the following:

* [UKFpython](https://bitbucket.org/berkeleylab/eetd-estcp_ndw_eis/src/e6587e567beec08106400fdaeeed9030294ca2c6/UKFpython?at=master)
	This contains the implementation of the Unscented Kalman Filter for python. This can be used for state and parameter estimation of a dynamic system given noisy measurements.


* [managementMeasurementsPy](https://bitbucket.org/berkeleylab/eetd-estcp_ndw_eis/src/e6587e567beec08106400fdaeeed9030294ca2c6/manageMeasurementsPy?at=master)
	This contains the program that allows to take data from external .csv files and putting them into the DB. S
	Several operations can be performed accessing the DB (e.g. plotting of time series, statistics about sensors and measurements, export time series to Modelica, etc.)


* [chillerMeasurements](https://bitbucket.org/berkeleylab/eetd-estcp_ndw_eis/src/e6587e567beec08106400fdaeeed9030294ca2c6/chillerMeasurements?at=master)
	Simple script that convert a collection of measurements contained into a .csv file, and create a .txt containing a table that can be used in Modelica.

* [modelicaPY](https://bitbucket.org/berkeleylab/eetd-estcp_ndw_eis/src/8fe4430768ed56ad81d51e675fe5ba6b07af31a7/modelicaPY?at=master)
	Collection of examples for interfacing modelica models with python. Have been used both BuildingsPy and PyFMI.





