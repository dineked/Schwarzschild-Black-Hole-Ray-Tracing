import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from typing import Optional

from constants import M, RS, RTOL, ATOL, MAX_STEP, HORIZON_EPS, R_ESCAPE

# Numerical tolerances used near radial turning points.
TURN_TOL = 1e-6
RESTART_TOL = 1e-4


def f(r: float) -> float:
    """
    Schwarzschild lapse factor.

    In geometric units with G = c = 1 and RS = 2M,

        f(r) = 1 - 2M/r.
    """
    return 1.0 - RS / r


def critical_impact_parameter() -> float:
    """
    Critical impact parameter for Schwarzschild null geodesics.

    For a Schwarzschild black hole,

        b_crit = 3 sqrt(3) M.
    """
    return 3.0 * np.sqrt(3.0) * M


def radial_radicand(r: float, b: float) -> float:
    """
    Radial radicand for the reduced equatorial null-geodesic equation.

    I use E = 1 and L = b, so the radial equation is

        (dr/dlambda)^2 = 1 - f(r) b^2 / r^2.

    The quantity returned here is the right-hand side,

        R(r; b) = 1 - f(r) b^2 / r^2.
    """
    return 1.0 - f(r) * (b**2) / (r**2)


def initial_ur(r0: float, b: float, inward: bool = True) -> float:
    """
    Construct the initial radial derivative dr/dlambda.

    This is not evolved as a separate variable in the reduced solver.
    It is useful as a check of the initial null condition.
    """
    rad = radial_radicand(r0, b)

    if rad < 0:
        raise ValueError(
            f"No real null geodesic for r0={r0}, b={b}: radicand={rad}"
        )

    ur = np.sqrt(rad)
    return -ur if inward else ur


def safe_sqrt_radicand(r: float, b: float) -> float:
    """
    Return sqrt(R(r; b)) with a small round-off guard.

    Near a radial turning point, the analytic value of R(r; b) approaches
    zero. Floating-point round-off can make it slightly negative. If the
    negative value is smaller than TURN_TOL, I set it to zero. Larger
    negative values are treated as unphysical.
    """
    rad = radial_radicand(r, b)

    if rad < 0 and abs(rad) < TURN_TOL:
        rad = 0.0
    elif rad < 0:
        raise ValueError(
            f"Negative radicand encountered: r={r}, b={b}, rad={rad}"
        )

    return np.sqrt(rad)


def branch_rhs_factory(b: float, sign_r: float):
    """
    Create the right-hand side for one radial branch.

    The reduced equations are

        dr/dlambda   = s_r sqrt(R(r; b)),
        dphi/dlambda = b/r^2,

    where s_r = -1 for the inward branch and s_r = +1 for the outward
    branch.

    I use separate branches because the reduced radial equation gives
    (dr/dlambda)^2. The sign of dr/dlambda must therefore be chosen
    explicitly.
    """
    if sign_r not in (-1.0, +1.0):
        raise ValueError("sign_r must be -1.0 for inward or +1.0 for outward.")

    def rhs(lam, Y):
        r, phi = Y

        dr_dlam = sign_r * safe_sqrt_radicand(r, b)
        dphi_dlam = b / (r**2)

        return np.array([dr_dlam, dphi_dlam], dtype=float)

    return rhs


def hit_horizon(lam, Y):
    """
    Terminal event for photon capture.

    The integration stops when

        r = RS + HORIZON_EPS = 2M + epsilon_horizon.

    This implements the thesis statement that captured rays are integrated
    inward until they reach the horizon cut-off.
    """
    r = Y[0]
    return r - (RS + HORIZON_EPS)


hit_horizon.terminal = True
hit_horizon.direction = -1


def reach_escape_radius(lam, Y):
    """
    Terminal event for photon escape.

    The integration stops when an outward-moving ray reaches R_ESCAPE.
    """
    r = Y[0]
    return r - R_ESCAPE


reach_escape_radius.terminal = True
reach_escape_radius.direction = +1


def reach_radius_event_factory(r_target: float):
    """
    Create a terminal event for reaching a specified radius from above.

    This is used for the inward branch of escaping rays. Instead of forcing
    the solver to step exactly onto the turning point, I stop at

        r_stop = r_turn + RESTART_TOL,

    then restart the outward branch from that radius.
    """
    def reach_radius(lam, Y):
        r = Y[0]
        return r - r_target

    reach_radius.terminal = True
    reach_radius.direction = -1

    return reach_radius


@dataclass
class EquatorialRayResult:
    """
    Container for one traced equatorial null geodesic.

    Attributes
    ----------
    b:
        Impact parameter of the ray.
    outcome:
        "capture", "escape", or "undecided".
    lam:
        Stored affine-parameter values.
    r:
        Stored Schwarzschild radial coordinate values.
    phi:
        Stored azimuthal coordinate values.
    turning_point_r:
        Analytic outer turning radius for escaping rays, if present.
    """
    b: float
    outcome: str
    lam: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    turning_point_r: Optional[float]


def integrate_branch(r0, phi0, b, sign_r, lam_max, events):
    """
    Integrate one radial branch over a finite affine-parameter interval.

    SciPy's solve_ivp requires a finite interval [0, lam_max]. Physical
    stopping conditions are imposed through terminal events. Therefore,
    when the thesis says that I integrate until the ray reaches the horizon
    cut-off or escape radius, this is implemented as event termination inside
    solve_ivp.
    """
    rhs = branch_rhs_factory(b, sign_r)

    sol = solve_ivp(
        rhs,
        (0.0, lam_max),
        np.array([r0, phi0], dtype=float),
        events=events,
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
    )

    return sol


def integrate_inward_until_horizon(r0, phi0, b, lam_max):
    """
    Integrate an inward ray until it reaches the horizon cut-off.

    The solver is called on [0, lam_max], but the terminal horizon event
    stops the integration early when

        r = 2M + HORIZON_EPS.

    If the event is not reached before lam_max, the calling function will
    classify the result as undecided.
    """
    return integrate_branch(
        r0=r0,
        phi0=phi0,
        b=b,
        sign_r=-1.0,
        lam_max=lam_max,
        events=[hit_horizon],
    )


def integrate_inward_until_turning_neighborhood(r0, phi0, b, r_stop, lam_max):
    """
    Integrate the inward branch of an escaping ray.

    For b > b_crit, the photon reaches an outer turning point. Numerically,
    I stop slightly outside that point at

        r_stop = r_turn + RESTART_TOL,

    because R(r; b) = 0 exactly at the turning point.
    """
    reach_turn_neighborhood = reach_radius_event_factory(r_stop)

    return integrate_branch(
        r0=r0,
        phi0=phi0,
        b=b,
        sign_r=-1.0,
        lam_max=lam_max,
        events=[reach_turn_neighborhood],
    )


def integrate_outward_until_escape(r0, phi0, b, lam_max):
    """
    Integrate the outward branch until the ray reaches the escape radius.

    The ray is classified as escaping if the terminal event

        r = R_ESCAPE

    is reached before the remaining affine-parameter interval ends.
    """
    return integrate_branch(
        r0=r0,
        phi0=phi0,
        b=b,
        sign_r=+1.0,
        lam_max=lam_max,
        events=[reach_escape_radius],
    )


def turning_radius(b: float) -> float:
    """
    Compute the outer radial turning point for an escaping ray.

    The turning point satisfies R(r; b) = 0. For Schwarzschild null
    geodesics this gives the cubic

        r^3 - b^2 r + 2 M b^2 = 0.

    I return the largest real root outside the event horizon.
    """
    coeffs = [1.0, 0.0, -b**2, 2.0 * M * b**2]
    roots = np.roots(coeffs)

    real_roots = [root.real for root in roots if abs(root.imag) < 1e-10]
    physical_roots = [r for r in real_roots if r > RS]

    if not physical_roots:
        raise ValueError(f"No physical turning radius found for b={b}")

    return max(physical_roots)


def has_turning_point(b: float) -> bool:
    """
    Return True if the ray has an outer turning point.

    For rays launched from large radius in Schwarzschild spacetime, this
    occurs for b > b_crit.
    """
    return b > critical_impact_parameter()


def trace_equatorial_ray(r0=30.0, b=6.0, lam_max=500.0):
    """
    Trace one equatorial Schwarzschild null geodesic.

    The reduced equations are

        dr/dlambda   = s_r sqrt(R(r; b)),
        dphi/dlambda = b/r^2.

    Strategy
    --------
    1. For b <= b_crit, integrate the inward branch until the horizon
       cut-off event is reached.

    2. For b > b_crit, compute the analytic turning radius, integrate
       inward to r_turn + RESTART_TOL, then restart on the outward branch
       until the escape-radius event is reached.

    In all cases, solve_ivp is called on a finite interval [0, lam_max].
    The physical stopping conditions are implemented using terminal events.
    """
    if r0 <= RS:
        raise ValueError("Initial radius must satisfy r0 > 2M.")

    bcrit = critical_impact_parameter()

    # ------------------------------------------------------------
    # Case 1: captured or critical branch.
    # ------------------------------------------------------------
    if b <= bcrit:
        sol_in = integrate_inward_until_horizon(
            r0=r0,
            phi0=0.0,
            b=b,
            lam_max=lam_max,
        )

        outcome = "capture" if sol_in.t_events[0].size > 0 else "undecided"

        return EquatorialRayResult(
            b=b,
            outcome=outcome,
            lam=np.array(sol_in.t),
            r=np.array(sol_in.y[0]),
            phi=np.array(sol_in.y[1]),
            turning_point_r=None,
        )

    # ------------------------------------------------------------
    # Case 2: escaping branch with a radial turning point.
    # ------------------------------------------------------------
    r_turn = turning_radius(b)
    r_stop = r_turn + RESTART_TOL

    sol_in = integrate_inward_until_turning_neighborhood(
        r0=r0,
        phi0=0.0,
        b=b,
        r_stop=r_stop,
        lam_max=lam_max,
    )

    if sol_in.t_events[0].size == 0:
        return EquatorialRayResult(
            b=b,
            outcome="undecided",
            lam=np.array(sol_in.t),
            r=np.array(sol_in.y[0]),
            phi=np.array(sol_in.y[1]),
            turning_point_r=float(r_turn),
        )

    lam_turn = sol_in.t_events[0][0]
    phi_turn = sol_in.y_events[0][0][1]

    lam_all = list(sol_in.t)
    r_all = list(sol_in.y[0])
    phi_all = list(sol_in.y[1])

    remaining_lam = max(lam_max - lam_turn, 1e-8)

    sol_out = integrate_outward_until_escape(
        r0=r_stop,
        phi0=phi_turn,
        b=b,
        lam_max=remaining_lam,
    )

    # Shift the outward affine-parameter values so that the full stored
    # trajectory has a continuous lambda array.
    lam_out = lam_turn + sol_out.t[1:]
    r_out = sol_out.y[0][1:]
    phi_out = sol_out.y[1][1:]

    lam_all.extend(lam_out)
    r_all.extend(r_out)
    phi_all.extend(phi_out)

    outcome = "escape" if sol_out.t_events[0].size > 0 else "undecided"

    return EquatorialRayResult(
        b=b,
        outcome=outcome,
        lam=np.array(lam_all),
        r=np.array(r_all),
        phi=np.array(phi_all),
        turning_point_r=float(r_turn),
    )


def trace_many_equatorial_rays(r0=30.0, b_values=None, lam_max=500.0):
    """
    Trace a list of equatorial rays for different impact parameters.
    """
    if b_values is None:
        b_values = np.linspace(4.8, 5.6, 25)

    results = []

    for b in b_values:
        try:
            result = trace_equatorial_ray(r0=r0, b=b, lam_max=lam_max)
            results.append(result)
        except Exception as e:
            print(f"b={b:.6f} failed: {e}")

    return results


def to_cartesian(result: EquatorialRayResult):
    """
    Convert the polar trajectory (r, phi) to plotting coordinates.

    These are equatorial-plane plotting coordinates, not observer-screen
    impact-parameter coordinates.
    """
    x = result.r * np.cos(result.phi)
    y = result.r * np.sin(result.phi)

    return x, y