within FmuExamples;
model ValveSimple
  extends ValveStuckBase(valve(filteredOpening=true));
equation
  connect(cmd, valve.opening) annotation (Line(
      points={{-80,60},{20,60},{20,18}},
      color={0,0,127},
      smooth=Smooth.None));
  annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-100,
            -100},{100,100}}), graphics));
end ValveSimple;
