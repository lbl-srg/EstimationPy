within FmuExamples.Examples;
model TestPumpA
  Modelica.Blocks.Sources.Ramp Rpm(duration=100, height=60)
    annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
  PumpA pump_2(
    a5=0.7,
    a7=0.8,
    a9=0.9) annotation (Placement(transformation(extent={{-12,-10},{8,10}})));
  PumpA pump_4(
    a5=0.1,
    a7=0.2,
    a9=0.3) annotation (Placement(transformation(extent={{-12,-80},{8,-60}})));
  PumpA pump_3(
    a5=0.2,
    a7=0.2,
    a9=0.2) annotation (Placement(transformation(extent={{-12,-40},{8,-20}})));
  PumpA pump_1(
    a5=0.5,
    a7=0.5,
    a9=0.8) annotation (Placement(transformation(extent={{-12,20},{8,40}})));
  PumpA pump_0 annotation (Placement(transformation(extent={{-12,60},{8,80}})));
equation
  connect(Rpm.y, pump_2.Nrpm) annotation (Line(
      points={{-39,0},{-12,0}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Rpm.y, pump_1.Nrpm) annotation (Line(
      points={{-39,0},{-26,0},{-26,30},{-12,30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Rpm.y, pump_0.Nrpm) annotation (Line(
      points={{-39,0},{-26,0},{-26,70},{-12,70}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Rpm.y, pump_3.Nrpm) annotation (Line(
      points={{-39,0},{-26,0},{-26,-30},{-12,-30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Rpm.y, pump_4.Nrpm) annotation (Line(
      points={{-39,0},{-26,0},{-26,-70},{-12,-70}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,
            -100},{100,100}}), graphics),
    experiment(StopTime=150),
    __Dymola_experimentSetupOutput);
end TestPumpA;
