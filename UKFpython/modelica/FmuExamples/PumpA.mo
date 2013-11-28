within FmuExamples;
model PumpA
  parameter Modelica.SIunits.MassFlowRate v_flow_nominal= 600
    "Nominal mass flow rate" annotation(fixed=False);
  parameter Modelica.SIunits.Power P_nom = 4.4 annotation(fixed=false);
  parameter Real RPM_nom = 60 annotation(fixed=false);
  parameter Real rpm_start = 0 annotation(fixed=false);
  parameter Real a(min=0.0, max = 1.0) = 0.0 annotation(fixed=false);
  parameter Real a1(min=1.0, max = 3.0) = 1.0 annotation(fixed=false);
  parameter Real b(min=0.0, max = 1.0) = 0.0 annotation(fixed=false);
  parameter Real b1(min=1.0, max = 3.0) = 1.0 annotation(fixed=false);

  Real x = motor.y / RPM_nom;

  Modelica.Blocks.Interfaces.RealInput Nrpm "Prescribed rotational speed"
    annotation (Placement(transformation(extent={{-120,-20},{-80,20}}),
        iconTransformation(extent={{-120,-20},{-80,20}})));
  Modelica.Blocks.Interfaces.RealOutput V_flow
    "Volume flow rate from port_a to port_b" annotation (Placement(
        transformation(extent={{80,60},{100,80}}), iconTransformation(extent={{80,
            60},{100,80}})));
  Modelica.Blocks.Interfaces.RealOutput P_el "Electrical power consumed"
    annotation (Placement(transformation(extent={{80,20},{100,40}}),
        iconTransformation(extent={{80,20},{100,40}})));

  Modelica.Blocks.Continuous.CriticalDamping motor(
    n=2,
    initType=Modelica.Blocks.Types.Init.InitialOutput,
    y_start=rpm_start,
    f=4)
    annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
  Modelica.Blocks.Sources.RealExpression realExpression(y=v_flow_nominal*(a*x^2 +
        (1 - a)*x)^a1)
    annotation (Placement(transformation(extent={{0,60},{20,80}})));
  Modelica.Blocks.Sources.RealExpression realExpression1(y=P_nom*(b*x^2 + (1 -
        b)*x)^b1)
    annotation (Placement(transformation(extent={{0,20},{20,40}})));
equation

  connect(Nrpm, motor.u) annotation (Line(
      points={{-100,1.11022e-15},{-81,1.11022e-15},{-81,0},{-62,0}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(realExpression1.y, P_el) annotation (Line(
      points={{21,30},{90,30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(realExpression.y, V_flow) annotation (Line(
      points={{21,70},{90,70}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,
            -100},{100,100}}), graphics), Icon(coordinateSystem(
          preserveAspectRatio=false, extent={{-100,-100},{100,100}}), graphics={
          Ellipse(extent={{-100,100},{100,-100}}, lineColor={0,0,125}), Polygon(
          points={{-60,80},{-60,-78},{100,0},{-60,80}},
          lineColor={0,0,255},
          smooth=Smooth.None,
          fillColor={0,0,127},
          fillPattern=FillPattern.Solid)}));
end PumpA;
