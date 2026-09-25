"""PhoenixVelocity: the 45-D velocity-conditioned GO2 locomotion task and its contract.

Import-light by design: :mod:`phoenix.velocity.contract` needs no torch or Isaac Lab, so the
deploy path and CI can use it. Isaac Lab code lives in submodules that import
Isaac Lab lazily.
"""
