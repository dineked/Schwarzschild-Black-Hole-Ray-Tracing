import numpy as np
import matplotlib.pyplot as plt

from constants import M, RS
from equatorial_solver import (
    trace_equatorial_ray,
    critical_impact_parameter,
    to_cartesian,
)


def total_angular_displacement(result):
    """
    Compute the total angular displacement along the stored trajectory.

    The stored trajectory contains phi(lambda), so the total angular change is

        Delta phi = |phi_final - phi_initial|.

    This quantity is useful for both captured and escaping rays. For escaping
    rays, it includes the inward and outward branches. For captured rays, it
    measures the angular displacement before the ray reaches the horizon
    cut-off.
    """
    return abs(result.phi[-1] - result.phi[0])


def approximate_winding_number(result):
    """
    Estimate the amount of angular winding around the black hole.

    I define

        N ≈ Delta phi / (2 pi).

    This is not an exact count of circular photon orbits at r = 3M. It is a
    numerical diagnostic for how much angular displacement the ray accumulates.
    Near b_crit, this quantity grows because the photon spends more time near
    the unstable photon sphere before it escapes or is captured.
    """
    return total_angular_displacement(result) / (2.0 * np.pi)


def representative_impact_parameters():
    """
    Impact parameters used for the representative shadow-boundary ray plot.

    The values include direct capture, near-critical capture, near-critical
    escape, and ordinary escape. The pair b = 5.1960 and b = 5.1962 lies on
    opposite sides of

        b_crit = 3 sqrt(3) M,

    and illustrates the capture--escape transition.
    """
    return np.array([
        0.0,
        2.0,
        4.0,
        5.0,
        5.18,
        5.195,
        5.1960,
        5.1962,
        5.20,
        5.4,
        6.0,
    ])


def trace_representative_rays(r0=30.0, lam_max=4000.0):
    """
    Trace the representative rays used in the shadow-boundary figure.

    The actual ray integration and capture--escape classification are handled
    in equatorial_solver.py. This function only selects the impact parameters
    and stores the resulting trajectories.
    """
    b_values = representative_impact_parameters()

    return [
        trace_equatorial_ray(
            r0=r0,
            b=b,
            lam_max=lam_max,
        )
        for b in b_values
    ]


def plot_shadow_with_representative_rays(
    r0=30.0,
    lam_max=4000.0,
    save=False,
    filename="shadow_with_rays.png",
):
    """
    Plot representative null geodesics near the shadow boundary.

    This figure is drawn in Schwarzschild equatorial coordinate space using

        x = r cos(phi),
        y = r sin(phi).

    It is not the observer-screen shadow grid. Instead, it shows the trajectory
    origin of the shadow boundary: rays below b_crit are captured, while rays
    above b_crit reach a turning point and escape.

    The event horizon at r = 2M is drawn as a black disk. The photon sphere at
    r = 3M is drawn as a dotted circle for reference.
    """
    bcrit = critical_impact_parameter()
    results = trace_representative_rays(r0=r0, lam_max=lam_max)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Event horizon at r = 2M.
    horizon = plt.Circle(
        (0, 0),
        RS,
        color="black",
        zorder=10,
        label=r"event horizon $r=2M$",
    )
    ax.add_patch(horizon)

    # Photon sphere at r = 3M. This is a theoretical reference curve.
    photon_sphere = plt.Circle(
        (0, 0),
        3.0 * M,
        fill=False,
        linestyle=":",
        linewidth=2,
        label=r"photon sphere $r=3M$",
    )
    ax.add_patch(photon_sphere)

    for result in results:
        x, y = to_cartesian(result)

        if result.outcome == "capture":
            linestyle = "-"
        elif result.outcome == "escape":
            linestyle = "--"
        else:
            linestyle = "-."

        label = None

        # Label only the most near-critical rays to keep the legend readable.
        if abs(result.b - bcrit) < 0.005:
            label = rf"$b={result.b:.4f}$, {result.outcome}"

        ax.plot(
            x,
            y,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.85,
            label=label,
        )

    # Mark the common launch point used for these rays.
    ax.plot(
        r0,
        0,
        "ko",
        markersize=5,
        label=rf"launch point $r_0={r0:.0f}M$",
    )

    ax.set_aspect("equal", "box")
    ax.set_xlim(-8, 12)
    ax.set_ylim(-8, 8)

    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.set_title("Null geodesics near the Schwarzschild shadow boundary")
    ax.grid(True, alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=9)

    fig.tight_layout()

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    print_representative_ray_summary(results)


def print_representative_ray_summary(results):
    """
    Print the classification and approximate winding number for each ray.

    This supports the thesis discussion of near-critical rays accumulating
    larger angular displacement close to b_crit.
    """
    bcrit = critical_impact_parameter()

    print(f"Critical impact parameter b_crit = {bcrit:.12f}")
    print("Representative ray summary")
    print("--------------------------")

    for result in results:
        print(
            f"b={result.b:.6f} | "
            f"{result.outcome.upper():9s} | "
            f"Delta_phi={total_angular_displacement(result):.6f} | "
            f"N≈{approximate_winding_number(result):.3f}"
        )


def near_critical_impact_parameters(n=40):
    """
    Impact-parameter values used for the near-critical winding diagnostic.

    The interval is centred around the analytic critical value. It samples
    both captured and escaping rays near the capture--escape boundary.
    """
    return np.linspace(5.17, 5.22, n)


def compute_near_critical_winding(r0=30.0, lam_max=6000.0, n=40):
    """
    Trace near-critical rays and compute their approximate winding numbers.

    The winding number is computed from the stored angular displacement,

        N ≈ Delta phi / (2 pi).

    This quantity is used as a diagnostic of near-critical behaviour. It is
    not an exact orbital count.
    """
    b_values = near_critical_impact_parameters(n=n)

    turns = []
    outcomes = []

    for b in b_values:
        result = trace_equatorial_ray(
            r0=r0,
            b=b,
            lam_max=lam_max,
        )

        turns.append(approximate_winding_number(result))
        outcomes.append(result.outcome)

    return b_values, np.array(turns), np.array(outcomes)


def plot_orbit_count_near_critical(
    r0=30.0,
    lam_max=6000.0,
    n=40,
    save=False,
    filename="orbit_count_near_critical.png",
):
    """
    Plot approximate winding number near the critical impact parameter.

    The vertical dashed line marks

        b_crit = 3 sqrt(3) M.

    Rays close to this value can accumulate large angular displacement near
    the unstable photon sphere before they are finally captured or escape.
    """
    bcrit = critical_impact_parameter()

    b_values, turns, outcomes = compute_near_critical_winding(
        r0=r0,
        lam_max=lam_max,
        n=n,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    for outcome in ["capture", "escape", "undecided"]:
        mask = outcomes == outcome

        if np.any(mask):
            ax.scatter(
                b_values[mask],
                turns[mask],
                label=outcome,
                s=50,
            )

    ax.axvline(
        bcrit,
        linestyle="--",
        linewidth=1.5,
        label=rf"$b_{{\rm crit}}={bcrit:.6f}$",
    )

    ax.set_xlabel(r"Impact parameter $b/M$")
    ax.set_ylabel(r"Approximate winding number $N \approx \Delta\phi/(2\pi)$")
    ax.set_title("Near-critical angular winding")
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.tight_layout()

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    """
    Generate the representative ray and winding figures used in the thesis.

    These figures connect the capture--escape classification to the idealised
    Schwarzschild shadow boundary. They show that near-critical rays pass close
    to the photon sphere and accumulate larger angular displacement.
    """
    plot_shadow_with_representative_rays(save=True)
    plot_orbit_count_near_critical(save=True)


if __name__ == "__main__":
    main()