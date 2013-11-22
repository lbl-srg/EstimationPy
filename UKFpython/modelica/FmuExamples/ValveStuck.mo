within FmuExamples;
model ValveStuck
  extends FmuExamples.ValveStuckBase;
  parameter Real leakPosition(min=0.0, max=1.0) = 0.2;
  parameter Real stuckPosition(min=0.0, max=1.0) = 0.8;
  Modelica.Blocks.Nonlinear.VariableLimiter
                                    limiter
    annotation (Placement(transformation(extent={{-20,50},{0,70}})));
  Modelica.Blocks.Sources.RealExpression Max(y=stuckPosition)
    annotation (Placement(transformation(extent={{-60,70},{-40,90}})));
  Modelica.Blocks.Sources.RealExpression Min(y=leakPosition)
    annotation (Placement(transformation(extent={{-60,30},{-40,50}})));

equation
  connect(Max.y, limiter.limit1) annotation (Line(
      points={{-39,80},{-32,80},{-32,68},{-22,68}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(Min.y, limiter.limit2) annotation (Line(
      points={{-39,40},{-30,40},{-30,52},{-22,52}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(cmd, limiter.u) annotation (Line(
      points={{-80,60},{-22,60}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(limiter.y, valve.opening) annotation (Line(
      points={{1,60},{20,60},{20,18}},
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
