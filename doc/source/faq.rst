Frequenty Asked Questions
=========================

For any questions about **EstimationPy** please contact

 * Marco Bonvini MBonvini@lbl.gov

If you want to contribute to the package please create a separate branch with
your proposed feature and make a pull request. 


**How to run examples and generate plots with results if a X server is not available?**

If you're using EstimationPy in a Docker container and you don't have linked its
X server to the one of the host OS, you can use the following shorcut.
Open a terminal and type::

   python -c 'import matplotlib; \
   matplotlib.use("Agg");\
   from estimationpy.examples.stuck_valve import run_ukf_smooth_fdd; \
   run_ukf_smooth_fdd.main()'

The former set of instructions loads **matplotlib**, defines the operation
mode that allows to save images without an X server, loads one of the
examples provided by EstimationPy and runs it.
