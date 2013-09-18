within StateEstimationIdentificationFMU;
model test



  Modelica.Blocks.Sources.Step Hot_Mflow(
    height=0.2,
    offset=0.5,
    startTime=360)
    annotation (Placement(transformation(extent={{-100,60},{-80,80}})));
  Modelica.Blocks.Sources.Ramp HotAirTemp(
    height=20,
    duration=50,
    startTime=30,
    offset=273.15 + 20)
    annotation (Placement(transformation(extent={{-100,20},{-80,40}})));
  Modelica.Blocks.Sources.Step Cold_Mflow(
    offset=0.3,
    height=0.5,
    startTime=650)
    annotation (Placement(transformation(extent={{-100,-80},{-80,-60}})));
  Modelica.Blocks.Sources.Ramp ColdAirTemp(
    duration=50,
    startTime=30,
    height=-10,
    offset=273.15 + 20)
    annotation (Placement(transformation(extent={{-100,-40},{-80,-20}})));
  StateEstimationIdentificationFMU.HeatExchanger heatExchanger
    annotation (Placement(transformation(extent={{-20,-20},{20,20}})));
equation
  for i in 1:10 loop
  end for;
  for i in 1:10 loop
  end for;
  connect(Hot_Mflow.y, heatExchanger.mFlow_hot) annotation (Line(
      points={{-79,70},{-44,70},{-44,16},{-20,16}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(HotAirTemp.y, heatExchanger.T_hot) annotation (Line(
      points={{-79,30},{-48,30},{-48,5.6},{-20,5.6}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(ColdAirTemp.y, heatExchanger.T_cold) annotation (Line(
      points={{-79,-30},{-48,-30},{-48,-6},{-20,-6}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Cold_Mflow.y, heatExchanger.mFlow_cold) annotation (Line(
      points={{-79,-70},{-44,-70},{-44,-16},{-20,-16}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (                               Diagram(coordinateSystem(
          preserveAspectRatio=false, extent={{-100,-100},{100,100}}), graphics));
end test;
