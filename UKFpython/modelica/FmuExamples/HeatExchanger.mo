within FmuExamples;
model HeatExchanger "Model of a simple Heat Exchanger"
  parameter Modelica.SIunits.ThermalConductance G_hot = 200
    "Thermal conductance on the hot side";
  parameter Modelica.SIunits.ThermalConductance G_cold = 200
    "Thermal conductance on the hot side";
  parameter Modelica.SIunits.HeatCapacity Cm = 500 "Heat capacity of the metal";
  parameter Modelica.SIunits.Temperature Tm_start = 273.15 + 20
    "Initial tenmperature of the metal";
  parameter Modelica.SIunits.Temperature Thot_start = 273.15 + 20
    "Initial tenmperature of the hot water in the pipe";
  parameter Modelica.SIunits.Temperature Tcold_start = 273.15 + 20
    "Initial tenmperature of the cold water in the pipe";
  parameter Real mFlow_hot_start = 0.0
    "Initial value of the hot water mass flow rate";
  parameter Real mFlow_cold_start = 0.0
    "Initial value of the cold water mass flow rate";
  Modelica.Fluid.Sources.MassFlowSource_T HotWater(
    use_m_flow_in=true,
    use_T_in=true,
    nPorts=1,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{-66,40},{-46,60}})));
  Modelica.Fluid.Sources.MassFlowSource_T ColdWater(
    use_m_flow_in=true,
    use_T_in=true,
    nPorts=1,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{70,-20},{50,0}})));
  Modelica.Fluid.Sources.FixedBoundary hotSink(nPorts=1, redeclare package
      Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{90,40},{70,60}})));
  Modelica.Fluid.Sources.FixedBoundary coldSink(nPorts=1, redeclare package
      Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{-90,-20},{-70,0}})));
  Modelica.Fluid.Pipes.DynamicPipe hotPipe(
    length=5,
    diameter=0.1,
    use_HeatTransfer=true,
    useLumpedPressure=false,
    useInnerPortProperties=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.LocalPipeFlowHeatTransfer,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater,
    nNodes=Nslice,
    use_T_start=true,
    T_start=Thot_start)
    annotation (Placement(transformation(extent={{-10,60},{10,40}})));

  Modelica.Fluid.Pipes.DynamicPipe coldPipe(
    length=5,
    diameter=0.1,
    use_HeatTransfer=true,
    useLumpedPressure=false,
    useInnerPortProperties=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.LocalPipeFlowHeatTransfer,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater,
    nNodes=Nslice,
    use_T_start=true,
    T_start=Tcold_start)
    annotation (Placement(transformation(extent={{10,-20},{-10,0}})));

  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor metal(C=Cm, T(start=
          Tm_start))
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-22,20})));
  inner Modelica.Fluid.System system
    annotation (Placement(transformation(extent={{-60,10},{-40,30}})));
  Modelica.Fluid.Sensors.TemperatureTwoPort hotWater_OUT(redeclare package
      Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{20,40},{40,60}})));
  Modelica.Fluid.Sensors.TemperatureTwoPort coldWater_OUT(redeclare package
      Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{-16,0},{-36,-20}})));
  Modelica.Fluid.Sensors.TemperatureTwoPort hotWater_IN(redeclare package
      Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{-36,40},{-16,60}})));
  Modelica.Fluid.Sensors.TemperatureTwoPort coldWater_IN(redeclare package
      Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{40,0},{20,-20}})));
  Modelica.Fluid.Sensors.MassFlowRate mFlow_Cold(redeclare package Medium =
        Modelica.Media.Water.ConstantPropertyLiquidWater) annotation (Placement(
        transformation(
        extent={{10,10},{-10,-10}},
        rotation=0,
        origin={-52,-10})));
  Modelica.Fluid.Sensors.MassFlowRate mFlow_Hot(redeclare package Medium =
        Modelica.Media.Water.ConstantPropertyLiquidWater) annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={56,50})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor Ghot[Nslice](each G=G_hot/
        Nslice)
    annotation (Placement(transformation(
        extent={{-6,-6},{6,6}},
        rotation=90,
        origin={2.6628e-16,30})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor Gcold[Nslice](each G=G_cold
        /Nslice)
    annotation (Placement(transformation(
        extent={{-6,-6},{6,6}},
        rotation=270,
        origin={4.44089e-16,10})));
  Modelica.Blocks.Interfaces.RealInput T_hot "Temperature of hot water"
    annotation (Placement(transformation(extent={{-120,8},{-80,48}}),
        iconTransformation(extent={{-120,8},{-80,48}})));
  Modelica.Blocks.Interfaces.RealInput mFlow_hot "Mass flow rate of hot water"
    annotation (Placement(transformation(extent={{-120,60},{-80,100}})));
  Modelica.Blocks.Interfaces.RealInput T_cold "Temperature of cold water"
    annotation (Placement(transformation(extent={{-120,-50},{-80,-10}}),
        iconTransformation(extent={{-120,-50},{-80,-10}})));
  Modelica.Blocks.Interfaces.RealInput mFlow_cold
    "Mass flow rate of cold water"
    annotation (Placement(transformation(extent={{-120,-100},{-80,-60}})));
  Modelica.Blocks.Interfaces.RealOutput Thot_IN
    "Temperature of entering hot water"
    annotation (Placement(transformation(extent={{90,80},{110,100}}),
        iconTransformation(extent={{90,80},{110,100}})));
  Modelica.Blocks.Interfaces.RealOutput Thot_OUT
    "Temperature of leaving hot water"
    annotation (Placement(transformation(extent={{90,50},{110,70}}),
        iconTransformation(extent={{90,50},{110,70}})));
  Modelica.Blocks.Interfaces.RealOutput mFlow_HOT "Hot water flow rate"
                                           annotation (Placement(transformation(
          extent={{90,20},{110,40}}), iconTransformation(extent={{90,20},{110,40}})));
  Modelica.Blocks.Interfaces.RealOutput mFlow_COLD
    "Mass flow rate from of cold water"    annotation (Placement(transformation(
          extent={{90,-100},{110,-80}}), iconTransformation(extent={{90,-100},{110,
            -80}})));
  Modelica.Blocks.Interfaces.RealOutput Tcold_OUT
    "Temperature of leaving cold water"
    annotation (Placement(transformation(extent={{90,-70},{110,-50}}),
        iconTransformation(extent={{90,-70},{110,-50}})));
  Modelica.Blocks.Interfaces.RealOutput Tcold_IN
    "Temperature of entering cold water"
    annotation (Placement(transformation(extent={{90,-40},{110,-20}})));
protected
  parameter Integer Nslice = 10 "Number of volumes in the pipes";
  Modelica.Blocks.Continuous.FirstOrder firstOrder(
    k=1,
    T=0.1,
    initType=Modelica.Blocks.Types.Init.InitialOutput,
    y_start=mFlow_hot_start)
    annotation (Placement(transformation(extent={{-74,74},{-62,86}})));
  Modelica.Blocks.Continuous.FirstOrder firstOrder1(
    k=1,
    T=0.1,
    initType=Modelica.Blocks.Types.Init.InitialOutput,
    y_start=mFlow_cold_start)
    annotation (Placement(transformation(extent={{-74,-86},{-62,-74}})));
equation
  connect(hotPipe.port_b, hotWater_OUT.port_a) annotation (Line(
      points={{10,50},{20,50}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(coldPipe.port_b, coldWater_OUT.port_a) annotation (Line(
      points={{-10,-10},{-16,-10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(ColdWater.ports[1], coldWater_IN.port_a) annotation (Line(
      points={{50,-10},{40,-10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(coldWater_IN.port_b, coldPipe.port_a) annotation (Line(
      points={{20,-10},{10,-10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(HotWater.ports[1], hotWater_IN.port_a) annotation (Line(
      points={{-46,50},{-36,50}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(coldWater_OUT.port_b, mFlow_Cold.port_a) annotation (Line(
      points={{-36,-10},{-42,-10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(mFlow_Cold.port_b, coldSink.ports[1]) annotation (Line(
      points={{-62,-10},{-70,-10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(hotWater_OUT.port_b, mFlow_Hot.port_a) annotation (Line(
      points={{40,50},{46,50}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(mFlow_Hot.port_b, hotSink.ports[1]) annotation (Line(
      points={{66,50},{70,50}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(hotWater_IN.port_b, hotPipe.port_a) annotation (Line(
      points={{-16,50},{-10,50}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(Ghot.port_b, hotPipe.heatPorts) annotation (Line(
      points={{0,36},{0,45.6},{0.1,45.6}},
      color={191,0,0},
      smooth=Smooth.None));
  connect(Gcold.port_b, coldPipe.heatPorts) annotation (Line(
      points={{0,4},{0,-5.6},{-0.1,-5.6}},
      color={191,0,0},
      smooth=Smooth.None));
  for i in 1:Nslice loop
    connect(metal.port, Ghot[i].port_a) annotation (Line(
      points={{-12,20},{0,20},{0,24}},
      color={191,0,0},
      smooth=Smooth.None));
    connect(Gcold[i].port_a, metal.port) annotation (Line(
      points={{0,16},{0,20},{-12,20}},
      color={191,0,0},
      smooth=Smooth.None));
  end for;
  connect(HotWater.T_in, T_hot) annotation (Line(
      points={{-68,54},{-76,54},{-76,28},{-100,28}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(ColdWater.T_in, T_cold) annotation (Line(
      points={{72,-6},{82,-6},{82,-30},{-100,-30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(hotWater_IN.T, Thot_IN)
                             annotation (Line(
      points={{-26,61},{-26,90},{100,90}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(hotWater_OUT.T, Thot_OUT)
                              annotation (Line(
      points={{30,61},{30,60},{100,60}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(mFlow_Hot.m_flow, mFlow_HOT)
                                     annotation (Line(
      points={{56,61},{56,30},{100,30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(mFlow_Cold.m_flow, mFlow_COLD)
                                      annotation (Line(
      points={{-52,-21},{-52,-90},{100,-90}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(coldWater_OUT.T, Tcold_OUT)
                               annotation (Line(
      points={{-26,-21},{-26,-60},{100,-60}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(coldWater_IN.T, Tcold_IN)
                              annotation (Line(
      points={{30,-21},{30,-30},{100,-30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(mFlow_hot, firstOrder.u) annotation (Line(
      points={{-100,80},{-75.2,80}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(firstOrder.y, HotWater.m_flow_in) annotation (Line(
      points={{-61.4,80},{-54,80},{-54,68},{-82,68},{-82,58},{-66,58}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(mFlow_cold, firstOrder1.u) annotation (Line(
      points={{-100,-80},{-75.2,-80}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(firstOrder1.y, ColdWater.m_flow_in) annotation (Line(
      points={{-61.4,-80},{86,-80},{86,-2},{70,-2}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (                               Diagram(coordinateSystem(
          preserveAspectRatio=false, extent={{-100,-100},{100,100}}), graphics),
    Icon(coordinateSystem(preserveAspectRatio=false, extent={{-100,-100},{100,100}}),
        graphics={Rectangle(
          extent={{-100,100},{100,-100}},
          lineColor={0,0,0},
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid),
        Rectangle(
          extent={{-66,76},{76,44}},
          lineColor={0,0,0},
          fillColor={170,213,255},
          fillPattern=FillPattern.Solid),
        Rectangle(
          extent={{-64,-44},{78,-76}},
          lineColor={0,0,0},
          fillColor={0,128,255},
          fillPattern=FillPattern.Solid),
        Polygon(
          points={{-46,40},{-46,-22},{-60,-22},{-40,-42},{-20,-22},{-34,-22},{-34,
              40},{-46,40}},
          lineColor={0,0,0},
          smooth=Smooth.None,
          fillColor={255,85,85},
          fillPattern=FillPattern.Solid),
        Polygon(
          points={{48,40},{48,-22},{34,-22},{54,-42},{74,-22},{60,-22},{60,40},{
              48,40}},
          lineColor={0,0,0},
          smooth=Smooth.None,
          fillColor={255,85,85},
          fillPattern=FillPattern.Solid), Text(
          extent={{-94,26},{106,-16}},
          lineColor={0,0,0},
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid,
          textString="Heat
exchanger")}));
end HeatExchanger;
