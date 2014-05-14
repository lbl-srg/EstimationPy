within FmuExamples;
model ValveSimple2
  parameter Real Kv=m_flow_nominal*3600/1000
    "Kv (metric) flow coefficient [m3/h]" annotation(fixed=false);
  parameter Modelica.SIunits.Time riseTime=3 "Time constant of the filter";
  parameter Modelica.Media.Interfaces.PartialMedium.MassFlowRate m_flow_nominal=
     1.5 "Nominal mass flowrate";
  parameter Modelica.SIunits.Pressure dp_nominal=0.5*101325
    "Nominal pressure drop";
  parameter Real bias = 0 "biasing of mass flow rate sensor";
  parameter Real lambda = 0
    "scale factor of mass flow rate sensor [Delta Mass flow rate/ Delta T]";
  parameter Real Tref = 273.15 + 20 "Reference temperature for thermal drift";

  Buildings.Fluid.Actuators.Valves.TwoWayQuickOpening
                                            valve(
    CvData=Modelica.Fluid.Types.CvTypes.Kv,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater,
    Kv=Kv,
    m_flow_nominal=m_flow_nominal,
    riseTime=riseTime,
    filteredOpening=true)
    annotation (Placement(transformation(extent={{10,0},{30,20}})));

  Modelica.Fluid.Sources.Boundary_pT Source(
    nPorts=1,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater,
    use_p_in=true,
    use_T_in=true)
    annotation (Placement(transformation(extent={{-20,0},{0,20}})));

  Modelica.Fluid.Sources.Boundary_pT   Sink(
    nPorts=1,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater,
    use_p_in=true,
    p(displayUnit="Pa"))
    annotation (Placement(transformation(extent={{84,0},{64,20}})));

  Modelica.Fluid.Sensors.MassFlowRate mFlow(redeclare package Medium =
        Modelica.Media.Water.ConstantPropertyLiquidWater) annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={50,10})));
  inner Modelica.Fluid.System system
    annotation (Placement(transformation(extent={{60,60},{80,80}})));
  Modelica.Blocks.Sources.RealExpression Pref(y=101325)
    annotation (Placement(transformation(extent={{-100,-70},{-80,-50}})));
  Modelica.Blocks.Math.Add add
    annotation (Placement(transformation(extent={{-60,0},{-40,20}})));
  Modelica.Blocks.Interfaces.RealInput cmd "Valve command" annotation (
      Placement(transformation(extent={{-100,40},{-60,80}}), iconTransformation(
        extent={{-20,-20},{20,20}},
        rotation=270,
        origin={0,100})));
  Modelica.Blocks.Interfaces.RealInput dp "Pressure difference" annotation (
      Placement(transformation(extent={{-120,-52},{-80,-12}}),
        iconTransformation(extent={{-120,-52},{-80,-12}})));
  Modelica.Blocks.Interfaces.RealOutput m_flow "Mass flow rate from"
    annotation (Placement(transformation(extent={{100,-30},{120,-10}}),
        iconTransformation(extent={{100,-30},{120,-10}})));
  Modelica.Blocks.Interfaces.RealOutput m_flow_real "Mass flow rate from"
    annotation (Placement(transformation(extent={{100,10},{120,30}}),
        iconTransformation(extent={{100,10},{120,30}})));
  Modelica.Blocks.Interfaces.RealInput T_in "Prescribed boundary temperature"
    annotation (Placement(transformation(extent={{-120,10},{-80,50}}),
        iconTransformation(extent={{-120,10},{-80,50}})));
equation

  m_flow = (1+lambda*(T_in - Tref))*mFlow.m_flow + bias;

  connect(Source.ports[1], valve.port_a) annotation (Line(
      points={{4.44089e-16,10},{10,10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(valve.port_b, mFlow.port_a) annotation (Line(
      points={{30,10},{40,10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(mFlow.port_b, Sink.ports[1]) annotation (Line(
      points={{60,10},{64,10}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(Pref.y, Sink.p_in) annotation (Line(
      points={{-79,-60},{96,-60},{96,18},{86,18}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(add.y, Source.p_in) annotation (Line(
      points={{-39,10},{-28,10},{-28,18},{-22,18}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Pref.y, add.u2) annotation (Line(
      points={{-79,-60},{-68,-60},{-68,4},{-62,4}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(dp, add.u1) annotation (Line(
      points={{-100,-32},{-79,-32},{-79,16},{-62,16}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(mFlow.m_flow, m_flow_real) annotation (Line(
      points={{50,21},{50,40},{96,40},{96,20},{110,20}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Source.T_in, T_in) annotation (Line(
      points={{-22,14},{-26,14},{-26,30},{-100,30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(cmd, valve.y) annotation (Line(
      points={{-80,60},{20,60},{20,22}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,
            -100},{100,100}}), graphics), Icon(coordinateSystem(
          preserveAspectRatio=false, extent={{-100,-100},{100,100}}), graphics={
        Polygon(points={{-100,50},{100,-50},{100,50},{0,0},{-100,-50},{-100,50}},
                    lineColor={0,0,0}),
        Ellipse(visible=filteredOpening,
          extent={{-40,94},{40,14}},
          lineColor={0,0,127},
          fillColor={255,255,255},
          fillPattern=FillPattern.Solid),
        Line(visible=filteredOpening,
          points={{-20,25},{-20,63},{0,41},{20,63},{20,25}},
          color={0,0,0},
          smooth=Smooth.None,
          thickness=0.5),
        Polygon(
          points={{20,-70},{60,-85},{20,-100},{20,-70}},
          lineColor={0,128,255},
          smooth=Smooth.None,
          fillColor={0,128,255},
          fillPattern=FillPattern.Solid,
          visible=showDesignFlowDirection),
        Line(
          points={{55,-85},{-60,-85}},
          color={0,128,255},
          smooth=Smooth.None,
          visible=showDesignFlowDirection),
        Text(
          extent={{-160,-40},{-100,-80}},
          lineColor={0,0,127},
          textString="Dp"),
        Text(
          extent={{100,-20},{160,-60}},
          lineColor={0,0,127},
          textString="m_flow"),
        Text(
          extent={{100,60},{180,20}},
          lineColor={0,0,127},
          textString="m_flow_real"),
        Text(
          extent={{-160,80},{-100,40}},
          lineColor={0,0,127},
          textString="T")}));
end ValveSimple2;
