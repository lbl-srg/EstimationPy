within FmuExamples;
model PumpA
  parameter Real gpm_nom = 600;
  parameter Modelica.SIunits.MassFlowRate m_flow_nominal= gpm_nom/60*3.7854*1.0
    "Nominal mass flow rate" annotation(fixed=False);
  parameter Modelica.SIunits.VolumeFlowRate v_flow_nominal= gpm_nom/60*3.7854/1000;
  parameter Modelica.SIunits.Power P_nom = 4.3 annotation(fixed=false);
  parameter Real RPM_nom = 60 annotation(fixed=false);
  parameter Modelica.SIunits.Pressure dp_nominal = 1000
    "Nominal pressure difference";
  parameter Real a1(min=0.0, max=1.0) = 1.0 annotation(fixed=false);
  parameter Real a3(min=0.0, max=1.0) = 1.0 annotation(fixed=false);
  parameter Real a5(min=0.0, max=1.0) = 1.0 annotation(fixed=false);
  parameter Real a7(min=0.0, max=1.0) = 1.0 annotation(fixed=false);
  parameter Real a9(min=0.0, max=1.0) = 1.0 annotation(fixed=false);

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

  Buildings.Fluid.Movers.FlowMachine_Nrpm pump(redeclare package Medium =
        Modelica.Media.Water.ConstantPropertyLiquidWater,
    N_nominal=3600,
    use_powerCharacteristic=true,
    motorCooledByFluid=false,
    hydraulicEfficiency(r_V={1}, eta={1}),
    pressure(V_flow={0,v_flow_nominal,2*v_flow_nominal}, dp={2*dp_nominal,
          dp_nominal,0}),
    motorEfficiency(r_V={1}, eta={1.0}),
    power(V_flow=v_flow_nominal*{0.1,0.3,0.5,0.7,0.9,1}, P={a1,a3,a5,a7,a9,1.0}))
    annotation (Placement(transformation(extent={{-18,-10},{2,10}})));
  Buildings.Fluid.FixedResistances.FixedResistanceDpM dpSta1(
    m_flow_nominal=m_flow_nominal,
    dp_nominal=dp_nominal/2,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    "Pressure drop"
    annotation (Placement(transformation(extent={{-48,-10},{-28,10}})));
  Buildings.Fluid.FixedResistances.FixedResistanceDpM dpSta(
    m_flow_nominal=m_flow_nominal,
    dp_nominal=dp_nominal/2,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater)
    "Pressure drop"
    annotation (Placement(transformation(extent={{10,-10},{30,10}})));
  Buildings.Fluid.Sources.Boundary_pT sou(
    use_p_in=false,
    p=system.p_ambient,
    nPorts=2,
    redeclare package Medium = Modelica.Media.Water.ConstantPropertyLiquidWater,
    T=293.15) annotation (Placement(transformation(extent={{-80,-60},{-60,-40}},
          rotation=0)));

  inner Modelica.Fluid.System system
    annotation (Placement(transformation(extent={{60,-80},{80,-60}})));
  Buildings.Fluid.Sensors.VolumeFlowRate senVolFlo(m_flow_nominal=
        m_flow_nominal, redeclare package Medium =
        Modelica.Media.Water.ConstantPropertyLiquidWater)
    annotation (Placement(transformation(extent={{40,-10},{60,10}})));
  Modelica.Blocks.Math.Gain gain(k=60)
    annotation (Placement(transformation(extent={{-56,20},{-36,40}})));
  Modelica.Blocks.Math.Gain gain1(k=1000*60/3.7854)
    annotation (Placement(transformation(extent={{46,58},{66,78}})));
  Modelica.Blocks.Math.Gain gain2(k=P_nom)
    annotation (Placement(transformation(extent={{22,18},{42,38}})));
equation

  connect(dpSta1.port_b, pump.port_a) annotation (Line(
      points={{-28,0},{-18,0}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(pump.port_b, dpSta.port_a) annotation (Line(
      points={{2,0},{8,0},{8,4.44089e-16},{10,4.44089e-16}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(dpSta1.port_a, sou.ports[1]) annotation (Line(
      points={{-48,0},{-54,0},{-54,-48},{-60,-48}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(dpSta.port_b, senVolFlo.port_a) annotation (Line(
      points={{30,0},{40,0}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(senVolFlo.port_b, sou.ports[2]) annotation (Line(
      points={{60,0},{64,0},{64,-52},{-60,-52}},
      color={0,127,255},
      smooth=Smooth.None));
  connect(Nrpm, gain.u) annotation (Line(
      points={{-100,0},{-80,0},{-80,30},{-58,30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(gain.y, pump.Nrpm) annotation (Line(
      points={{-35,30},{-8,30},{-8,12}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(gain1.y, V_flow) annotation (Line(
      points={{67,68},{78,68},{78,70},{90,70}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(senVolFlo.V_flow, gain1.u) annotation (Line(
      points={{50,11},{50,44},{16,44},{16,68},{44,68}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(gain2.y, P_el) annotation (Line(
      points={{43,28},{66,28},{66,30},{90,30}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(pump.P, gain2.u) annotation (Line(
      points={{3,8},{10,8},{10,28},{20,28}},
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
