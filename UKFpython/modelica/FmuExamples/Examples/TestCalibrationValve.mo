within FmuExamples.Examples;
model TestCalibrationValve
  ValveStuckInputs valveStuck(                        bias=0, lambda=0.00)
    annotation (Placement(transformation(extent={{-10,-12},{10,8}})));
  Modelica.Blocks.Sources.Ramp Press(
    duration=50, height=101325*0.5)
    annotation (Placement(transformation(extent={{-60,-42},{-40,-22}})));
  Modelica.Blocks.Sources.Step leak(
    offset=0.0,
    height=0,
    startTime=0)
    annotation (Placement(transformation(extent={{-40,20},{-20,40}})));
  Modelica.Blocks.Sources.Step stuck(
    offset=1.0,
    height=0,
    startTime=0)
    annotation (Placement(transformation(extent={{-90,70},{-70,90}})));
  Modelica.Blocks.Sources.TimeTable cmd(
    offset=0,
    startTime=0,
    table=[0,1; 70,1; 100,0.1; 135,0; 150,0.2; 180,0.8; 240,1; 250,1])
    annotation (Placement(transformation(extent={{-60,50},{-40,70}})));
  Modelica.Blocks.Sources.Ramp Temp(
    duration=100,
    offset=273.15 + 15,
    height=30)
    annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
equation
  connect(Press.y, valveStuck.dp) annotation (Line(
      points={{-39,-32},{-18,-32},{-18,-5.2},{-10,-5.2}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(leak.y, valveStuck.leakPosition) annotation (Line(
      points={{-19,30},{-6,30},{-6,6}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(stuck.y, valveStuck.stuckPosition) annotation (Line(
      points={{-69,80},{6,80},{6,6}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(cmd.y, valveStuck.cmd) annotation (Line(
      points={{-39,60},{0,60},{0,8}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Temp.y, valveStuck.T_in) annotation (Line(
      points={{-59,0},{-34,0},{-34,1},{-10,1}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,-100},{
            100,100}}), graphics),
    experiment(StopTime=250),
    __Dymola_experimentSetupOutput);
end TestCalibrationValve;
