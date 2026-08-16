import numpy as np
import matplotlib.pyplot as plt

from constants import M, RS, HORIZON_EPS, R_ESCAPE
from equatorial_solver import (
    trace_equatorial_ray,
    trace_many_equatorial_rays,
    critical_impact_parameter,
    to_cartesian,
)
from shadow import plot_shadow, plot_shadow_angular_size


def print_numerical_setup():
    """
    Print the main physical and numerical settings used in this driver script.

    The actual ODE integration is implemented in equatorial_solver.py.
    This file selects impact parameters, calls the ray tracer, and produces
    the figures used to analyse the capture--escape transition.
    """
    bcrit = critical_impact_parameter()

    print("Numerical ray-tracing setup")
    print("---------------------------")
    print(f"M                  = {M}")
    print(f"Event horizon      = RS = 2M = {RS}")
    print(f"Horizon cut-off    = RS + HORIZON_EPS = {RS + HORIZON_EPS}")
    print(f"Escape radius      = R_ESCAPE = {R_ESCAPE}")
    print(f"Critical parameter = b_crit = {bcrit:.12f}")
    print()


def print_summary(results):
    """
    Print the capture--escape classification for a list of ray results.

    For escaping rays, the analytic outer turning radius is also printed.
    This corresponds to the branch-switching radius discussed in the thesis.
    """
    for result in results:
        turning_point_text = (
            f", r_turn={result.turning_point_r:.6f}"
            if result.turning_point_r is not None
            else ""
        )

        print(
            f"b={result.b:.6f}  "
            f"{result.outcome.upper()}"
            f"{turning_point_text}"
        )


def plot_one(result):
    """
    Plot one equatorial null geodesic in Schwarzschild coordinate space.

    The plotted coordinates are

        x = r cos(phi),
        y = r sin(phi).

    These are spatial plotting coordinates in the equatorial plane. They are
    not the observer-screen impact-parameter coordinates used for the shadow
    grid.
    """
    x, y = to_cartesian(result)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        x,
        y,
        linewidth=1.8,
        label=rf"$b={result.b:.6f}$, {result.outcome}",
    )

    horizon = plt.Circle(
        (0, 0),
        RS,
        color="black",
        zorder=10,
        label=r"event horizon $r=2M$",
    )
    ax.add_patch(horizon)

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.set_title("Single equatorial null geodesic")
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_many(results, zoom=False, title="Family of equatorial null geodesics"):
    """
    Plot a family of equatorial null geodesics.

    This function visualises the capture--escape classification described in
    the thesis. Each result has already been classified by the ray tracer:

        capture : the ray reached r = 2M + epsilon_horizon,
        escape  : the ray reached R_ESCAPE on the outward branch,
        undecided : neither event occurred before lam_max.

    The event horizon is drawn as a black disk at r = 2M.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    horizon = plt.Circle(
        (0, 0),
        RS,
        color="black",
        zorder=10,
        label=r"event horizon $r=2M$",
    )
    ax.add_patch(horizon)

    bcrit = critical_impact_parameter()

    b_values = np.array([result.b for result in results])
    b_min = np.min(b_values)
    b_max = np.max(b_values)

    for result in results:
        x, y = to_cartesian(result)

        label = None

        # Label only the near-critical subset so the legend remains readable.
        if abs(result.b - bcrit) < 0.03:
            label = rf"$b={result.b:.4f}$"

        ax.plot(
            x,
            y,
            linewidth=1.0,
            alpha=0.9,
            label=label,
        )

    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.set_title(title)

    if zoom:
        ax.set_xlim(-8, 12)
        ax.set_ylim(-8, 8)

    handles, labels = ax.get_legend_handles_labels()

    if labels:
        ax.legend(
            handles,
            labels,
            title=(
                rf"Full plotted range: {b_min:.2f} $\leq b \leq$ {b_max:.2f}"
                "\n"
                rf"$b_{{\rm crit}} = {bcrit:.4f}$"
                "\n"
                rf"Labelled subset: $|b-b_{{\rm crit}}|<0.03$"
            ),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            title_fontsize=9,
            frameon=True,
        )

    fig.tight_layout()
    plt.show()


def broad_impact_parameters():
    """
    Impact-parameter values used for the broad capture--escape family.

    These values cover direct capture, near-critical capture, near-critical
    escape, and ordinary weakly deflected escape.
    """
    return np.array([
        0.0,       # direct capture
        1.0,       # direct capture
        2.0,       # direct capture
        3.0,       # direct capture
        4.0,       # capture with bending
        4.8,       # stronger capture
        5.0,       # strong capture
        5.15,      # capture, closer to critical
        5.18,      # near-critical capture
        5.19,      # near-critical capture
        5.195,     # whirl then capture
        5.1960,    # very near-critical capture
        5.1962,    # very near-critical escape
        5.197,     # whirl then escape
        5.20,      # near-critical escape
        5.205,     # near-critical escape
        5.25,      # ordinary escape
        5.4,       # broader escape
        6.0,       # weakly bent escape
        7.0,       # weak deflection
        8.0,       # weak deflection
        9.0,       # weak deflection
        10.0,      # weak deflection
    ])


def near_critical_impact_parameters():
    """
    Impact-parameter values used for the near-critical trajectory family.

    These values lie close to

        b_crit = 3 sqrt(3) M.

    The pair b = 5.1960 and b = 5.1962 is used to show that a small change
    in b can switch the ray from capture to escape.
    """
    return np.array([
        5.18,
        5.19,
        5.195,
        5.1960,
        5.1962,
        5.197,
        5.20,
    ])


def run_broad_family():
    """
    Trace the broad family of rays shown in the capture--escape figure.

    The launch radius is r0 = 30M, as described in the thesis. The larger
    lam_max gives near-critical rays enough affine-parameter range to reach
    their terminal event.
    """
    return trace_many_equatorial_rays(
        r0=30.0,
        b_values=broad_impact_parameters(),
        lam_max=1500.0,
    )


def run_near_critical_family():
    """
    Trace the near-critical family of rays.

    I use a larger lam_max here because rays close to b_crit spend a longer
    angular interval near the photon sphere before their final outcome is
    clear.
    """
    return trace_many_equatorial_rays(
        r0=30.0,
        b_values=near_critical_impact_parameters(),
        lam_max=4000.0,
    )


def plot_representative_single_rays():
    """
    Plot individual rays used as checks and illustrations.

    These four examples represent ordinary escape, ordinary capture,
    near-critical capture, and near-critical escape.
    """
    examples = [
        (6.0, 800.0),       # ordinary escape
        (4.0, 800.0),       # ordinary capture
        (5.1960, 4000.0),   # near-critical capture
        (5.1962, 4000.0),   # near-critical escape
    ]

    for b, lam_max in examples:
        result = trace_equatorial_ray(
            r0=30.0,
            b=b,
            lam_max=lam_max,
        )
        plot_one(result)


def main():
    """
    Run the thesis ray-tracing experiments.

    This script does four things:

    1. It traces a broad family of equatorial null geodesics.
    2. It traces a near-critical family around b_crit.
    3. It plots the corresponding capture--escape trajectory figures.
    4. It calls the shadow plotting routines in impact-parameter space.

    The physical stopping rules are implemented inside equatorial_solver.py:
    captured rays stop at r = 2M + epsilon_horizon, and escaping rays stop
    at R_ESCAPE on the outward branch.
    """
    print_numerical_setup()

    results = run_broad_family()
    results_near_critical = run_near_critical_family()

    print("Broad-family classification")
    print("---------------------------")
    print_summary(results)
    print()

    plot_many(
        results,
        zoom=True,
        title=r"Broad capture--escape family of equatorial null geodesics",
    )

    plot_many(
        results,
        zoom=False,
        title=r"Broad capture--escape family of equatorial null geodesics",
    )

    plot_many(
        results_near_critical,
        zoom=False,
        title=r"Near-critical family of equatorial null geodesics",
    )

    plot_many(
        results_near_critical,
        zoom=True,
        title=r"Near-critical family of equatorial null geodesics",
    )

    # Shadow plots in observer-screen impact-parameter space.
    plot_shadow(
        screen_size=8.0,
        resolution=600,
        ring_width=0.05,
    )

    plot_shadow_angular_size(
        r_min=3.1,
        r_max=80.0,
        n=400,
    )

    # Optional individual-ray checks.
    plot_representative_single_rays()


if __name__ == "__main__":
    main()