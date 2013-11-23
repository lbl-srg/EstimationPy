within FmuExamples;
model ValveStuck
  extends FmuExamples.ValveStuckBase;
  parameter Real valveCmd(min=0.0, max=1.0) = 1.0;

  Modelica.Blocks.Math.Feedback sub
    annotation (Placement(transformation(extent={{40,30},{60,50}})));
  Modelica.Blocks.Interfaces.RealOutput err
    annotation (Placement(transformation(extent={{100,30},{120,50}})));
  Modelica.Blocks.Math.Product product1
    annotation (Placement(transformation(extent={{-20,30},{0,50}})));
  Modelica.Blocks.Continuous.Integrator command(initType=Modelica.Blocks.Types.Init.InitialOutput,
      y_start=valveCmd)
    annotation (Placement(transformation(extent={{-60,30},{-40,50}})));
  Modelica.Blocks.Sources.Constant const(k=0)
    annotation (Placement(transformation(extent={{-100,30},{-80,50}})));
equation
  connect(cmd, sub.u1) annotation (Line(
      points={{-80,60},{32,60},{32,40},{42,40}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(sub.y, err) annotation (Line(
      points={{59,40},{110,40}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(cmd, product1.u1) annotation (Line(
      points={{-80,60},{-30,60},{-30,46},{-22,46}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(product1.y, valve.opening) annotation (Line(
      points={{1,40},{20,40},{20,18}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(product1.y, sub.u2) annotation (Line(
      points={{1,40},{26,40},{26,26},{50,26},{50,32}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(command.y, product1.u2) annotation (Line(
      points={{-39,40},{-26,40},{-26,34},{-22,34}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(const.y, command.u) annotation (Line(
      points={{-79,40},{-62,40}},
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
