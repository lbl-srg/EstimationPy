within FmuExamples;
model FirstOrder "Model of a linear SISO first order system"
  output Real x(start = 2.0) "State variable of the system";
  input Real u "Input variable";
  output Real y "Output of the system";
  parameter Real a = -1 "A coefficient of the SISO linear system"
    annotation(Fixed=false);
  parameter Real b = 2.5 "B coefficient of the SISO linear system"
    annotation(Fixed=false);
  parameter Real c = 3 "C coefficient of the SISO linear system"
    annotation(Fixed=false);
  parameter Real d = 0.1 "D coefficient of the SISO linear system"
    annotation(Fixed=false);
equation
  der(x) = a*x + b*u;
  y = c*x + d*u;
end FirstOrder;
