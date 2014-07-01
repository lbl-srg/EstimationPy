within FmuExamples;
model ValveStuck
  extends FmuExamples.ValveStuckBase;
  parameter Real valveCmd(min=0.0, max=1.0) = 1.0;
  parameter Real use_cmd = 0;
  Modelica.Blocks.Math.Feedback sub
    annotation (Placement(transformation(extent={{34,40},{54,60}})));
  Modelica.Blocks.Interfaces.RealOutput err
    annotation (Placement(transformation(extent={{100,30},{120,50}})));
  Modelica.Blocks.Sources.Constant const(k=0)
    annotation (Placement(transformation(extent={{-70,34},{-50,54}})));
  Modelica.Blocks.Continuous.Integrator command(initType=Modelica.Blocks.Types.Init.InitialOutput,
      y_start=valveCmd)
    annotation (Placement(transformation(extent={{-36,34},{-16,54}})));
  Modelica.Blocks.Math.Add add1
    annotation (Placement(transformation(extent={{0,76},{8,84}})));
  Modelica.Blocks.Math.Product
                           add2
    annotation (Placement(transformation(extent={{-20,84},{-12,92}})));
  Modelica.Blocks.Math.Product
                           add3
    annotation (Placement(transformation(extent={{-20,70},{-12,78}})));
  Modelica.Blocks.Sources.RealExpression
                                   const1(y=use_cmd)
    annotation (Placement(transformation(extent={{-70,76},{-50,96}})));
  Modelica.Blocks.Sources.RealExpression
                                   const2(y=1 - use_cmd)
    annotation (Placement(transformation(extent={{-70,62},{-50,82}})));
equation
  connect(cmd, sub.u1) annotation (Line(
      points={{-80,60},{32,60},{32,50},{36,50}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(sub.y, err) annotation (Line(
      points={{53,50},{98,50},{98,40},{110,40}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(const.y, command.u) annotation (Line(
      points={{-49,44},{-38,44}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(cmd, add2.u1) annotation (Line(
      points={{-80,60},{-42,60},{-42,90.4},{-20.8,90.4}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(command.y, add3.u2) annotation (Line(
      points={{-15,44},{-10,44},{-10,66},{-32,66},{-32,71.6},{-20.8,71.6}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(add2.y, add1.u1) annotation (Line(
      points={{-11.6,88},{-6,88},{-6,82.4},{-0.8,82.4}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(add3.y, add1.u2) annotation (Line(
      points={{-11.6,74},{-6,74},{-6,77.6},{-0.8,77.6}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(add1.y, valve.opening) annotation (Line(
      points={{8.4,80},{20,80},{20,18}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(add1.y, sub.u2) annotation (Line(
      points={{8.4,80},{26,80},{26,34},{44,34},{44,42}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(const1.y, add2.u2) annotation (Line(
      points={{-49,86},{-36,86},{-36,85.6},{-20.8,85.6}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(const2.y, add3.u1) annotation (Line(
      points={{-49,72},{-34,72},{-34,76.4},{-20.8,76.4}},
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
          extent={{-160,-20},{-100,-60}},
          lineColor={0,0,127},
          textString="Dp"),
        Text(
          extent={{100,-20},{160,-60}},
          lineColor={0,0,127},
          textString="m_flow")}));
end ValveStuck;
