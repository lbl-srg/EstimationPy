within FmuExamples;
model ValveStuckInputs
  extends FmuExamples.ValveStuckBase;
  inner Modelica.Fluid.System system
    annotation (Placement(transformation(extent={{60,60},{80,80}})));
  Modelica.Blocks.Interfaces.RealInput leakPosition "leak position" annotation (
      Placement(transformation(extent={{-84,16},{-44,56}}),  iconTransformation(
        extent={{-20,-20},{20,20}},
        rotation=270,
        origin={-60,80})));

  Modelica.Blocks.Interfaces.RealInput stuckPosition "stuck position" annotation (
      Placement(transformation(extent={{-76,60},{-36,100}}), iconTransformation(
        extent={{-20,-20},{20,20}},
        rotation=270,
        origin={60,80})));
  Buildings.Utilities.Math.SmoothMax smoothMax(deltaX=0.01)
    annotation (Placement(transformation(extent={{4,32},{24,52}})));
  Buildings.Utilities.Math.SmoothMin smoothMin(deltaX=0.01)
    annotation (Placement(transformation(extent={{-28,56},{-8,76}})));
equation
  connect(cmd, smoothMin.u2) annotation (Line(
      points={{-80,60},{-30,60}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(smoothMin.y, smoothMax.u1) annotation (Line(
      points={{-7,66},{-2,66},{-2,48},{2,48}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(leakPosition, smoothMax.u2) annotation (Line(
      points={{-64,36},{2,36}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(stuckPosition, smoothMin.u1) annotation (Line(
      points={{-56,80},{-34,80},{-34,72},{-30,72}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(smoothMax.y, valve.opening) annotation (Line(
      points={{25,42},{32,42},{32,26},{20,26},{20,18}},
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
end ValveStuckInputs;
