"""Independent sim-to-sim validation of PhoenixVelocity policies in MuJoCo.

A candidate that is stable only in Isaac Lab must not advance to hardware. This
package runs the SAME deployable policy artifact in a second simulator with a
different contact solver and integrator, and compares what it does against the
Isaac Lab evaluation of the same scenarios.

Import-light by design: :mod:`phoenix.sim2sim.scenarios`,
:mod:`phoenix.sim2sim.dc_motor` and :mod:`phoenix.sim2sim.compare` are pure
Python / numpy so the Isaac side and CI can import them. Only
:mod:`phoenix.sim2sim.model` and :mod:`phoenix.sim2sim.mujoco_runner` touch
MuJoCo, and they import it lazily.
"""
