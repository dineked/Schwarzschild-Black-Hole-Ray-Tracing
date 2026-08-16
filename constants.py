"""
Numerical and physical constants used by the Schwarzschild ray tracer.

The calculation uses geometric units with

    G = c = M = 1.

All radii and impact parameters are therefore measured in units of the
black-hole mass M.
"""

# Black-hole mass scale.
M = 1.0

# Schwarzschild radius / event horizon radius.
RS = 2.0 * M

# Adaptive ODE solver tolerances used by solve_ivp.
RTOL = 1e-10
ATOL = 1e-12

# Maximum allowed affine-parameter step.
MAX_STEP = 0.005

# Captured rays are stopped slightly outside the horizon,
# at r = 2M + HORIZON_EPS.
HORIZON_EPS = 1e-5

# Escaping rays are stopped once they reach this radius on the outward branch.
R_ESCAPE = 100.0