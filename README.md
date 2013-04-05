# EETD - ESTCP NDW EIS
## Environmental Energy Technology Division
## ESTCP project for Fault Detection and Diagnosis
## Naval District Washington D.C.
## Energy Information System

This project contains different resources for the project, mainly python and Modelica code. The resources are the following:

* [UKFpython](src/UKFpython?at=master)
	This contains the implementation of the Unscented Kalman Filter for python. This can be used for state and parameter estimation of a dynamic system given noisy measurements.


* [managementMeasurementsPy](src/manageMeasurementsPy?at=master)
	This contains the program that allows to take data from external .csv files and putting them into the DB. S
	Several operations can be performed accessing the DB (e.g. plotting of time series, statistics about sensors and measurements, export time series to Modelica, etc.)


* [chillerMeasurements](src/chillerMeasurements?at=master)
	Simple script that convert a collection of measurements contained into a .csv file, and create a .txt containing a table that can be used in Modelica.





