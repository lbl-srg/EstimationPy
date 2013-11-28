within FmuExamples.Examples;
model TestStuckValve
  ValveStuckInputs valveStuck(bias=0.05, lambda=0.01)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Blocks.Sources.Ramp Press(
    duration=50,
    height=101325*3)
    annotation (Placement(transformation(extent={{-62,-40},{-42,-20}})));
  Modelica.Blocks.Sources.Step leak(
    height=0.2,
    offset=0.0,
    startTime=80)
    annotation (Placement(transformation(extent={{-40,20},{-20,40}})));
  Modelica.Blocks.Sources.Step stuck(
    startTime=100,
    height=-0.4,
    offset=1.0)
    annotation (Placement(transformation(extent={{-90,70},{-70,90}})));
  Modelica.Blocks.Sources.TimeTable cmd(
    table=[0,1; 70,1; 100,0.1; 150,0.2; 180,0.8; 240,1],
    offset=0,
    startTime=0)
    annotation (Placement(transformation(extent={{-60,50},{-40,70}})));
  Modelica.Blocks.Sources.Ramp Temp(
    duration=100,
    offset=273.15 + 15,
    height=30)
    annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
equation
  connect(Press.y, valveStuck.dp) annotation (Line(
      points={{-41,-30},{-18,-30},{-18,-3.2},{-10,-3.2}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(leak.y, valveStuck.leakPosition) annotation (Line(
      points={{-19,30},{-6,30},{-6,8}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(stuck.y, valveStuck.stuckPosition) annotation (Line(
      points={{-69,80},{6,80},{6,8}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(cmd.y, valveStuck.cmd) annotation (Line(
      points={{-39,60},{0,60},{0,10}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Temp.y, valveStuck.T_in) annotation (Line(
      points={{-59,0},{-34,0},{-34,3},{-10,3}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,-100},{
            100,100}}), graphics),
    experiment(StopTime=250),
    __Dymola_experimentSetupOutput);
end TestStuckValve;
