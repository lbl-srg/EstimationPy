within FmuExamples.Examples;
model testFirstOrder
  FmuExamples.FirstOrder system;
equation
  system.u = if time <=10 then 1 else 2;
  annotation (experiment(StopTime=20), __Dymola_experimentSetupOutput);
end testFirstOrder;
