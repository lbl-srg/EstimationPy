within FmuExamples.Examples;
model TestPump
  Modelica.Blocks.Sources.Ramp Rpm(duration=100, height=60)
    annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
  Pump pump(
    a1=3.0,
    a=0.5,
    P_nom=12) annotation (Placement(transformation(extent={{-12,-10},{8,10}})));
equation
  connect(Rpm.y, pump.Nrpm) annotation (Line(
      points={{-39,0},{-12,0}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,
            -100},{100,100}}), graphics));
end TestPump;
