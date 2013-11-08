within FmuExamples;
model FirstOrder
  "Model of a linear SISO first order system - Used to test how to couple FMUs with the UKF python"
  output Real x(start = 2.0) "State variable of the FO system";
  input Real u "Input variable";
  output Real y "Output of the system";
  parameter Real a = -1 "A coefficient of the SISO linear system";
  parameter Real b = 2.5 "B coefficient of the SISO linear system";
  parameter Real c = 3 "C coefficient of the SISO linear system";
  parameter Real d = 0.1 "D coefficient of the SISO linear system";
equation
  der(x) = a*x + b*u;
  y = c*x + d*u;
end FirstOrder;
