import matplotlib.pyplot as plt

from constants import RS
from equatorial_solver import (
    trace_equatorial_ray,
    critical_impact_parameter,
    to_cartesian,
)


def plot_nearcritical_side_by_side(
    r0=30.0,
    lam_max=4000.0,
    b_capture=5.1960,
    b_escape=5.1962,
    save=True,
    filename="fig_nearcritical_side_by_side.png",
):
    """
    Plot two near-critical Schwarzschild null geodesics side by side.

    This figure compares two rays with impact parameters very close to

        b_crit = 3 sqrt(3) M.

    The first ray has b < b_crit and is expected to be captured. The second
    ray has b > b_crit and is expected to escape after reaching a radial
    turning point.

    Both panels use the same axis limits, so the near-critical behaviour can
    be compared directly.
    """
    bcrit = critical_impact_parameter()

    result_cap = trace_equatorial_ray(
        r0=r0,
        b=b_capture,
        lam_max=lam_max,
    )

    result_esc = trace_equatorial_ray(
        r0=r0,
        b=b_escape,
        lam_max=lam_max,
    )

    x_cap, y_cap = to_cartesian(result_cap)
    x_esc, y_esc = to_cartesian(result_esc)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
        sharex=True,
        sharey=True,
    )

    # Common axis limits make the two trajectories directly comparable.
    x_min, x_max = -4.5, 8.0
    y_min, y_max = -5.0, 6.0

    # ------------------------------------------------------------
    # Left panel: near-critical captured trajectory.
    # ------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        x_cap,
        y_cap,
        linewidth=2.0,
        label=rf"$b={b_capture:.4f}$, {result_cap.outcome}",
    )

    horizon = plt.Circle(
        (0, 0),
        RS,
        color="black",
        zorder=10,
    )
    ax.add_patch(horizon)

    ax.set_title("Near-critical captured trajectory")
    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.4)
    ax.legend()

    # ------------------------------------------------------------
    # Right panel: near-critical escaping trajectory.
    # ------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        x_esc,
        y_esc,
        linewidth=2.0,
        label=rf"$b={b_escape:.4f}$, {result_esc.outcome}",
    )

    horizon = plt.Circle(
        (0, 0),
        RS,
        color="black",
        zorder=10,
    )
    ax.add_patch(horizon)

    ax.set_title("Near-critical escaping trajectory")
    ax.set_xlabel(r"$x/M$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.suptitle(
        rf"Near-critical trajectories around $b_{{\rm crit}}={bcrit:.6f}$",
        y=1.03,
    )

    fig.tight_layout()

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    print("Near-critical side-by-side comparison")
    print("-------------------------------------")
    print(f"b_crit = {bcrit:.12f}")
    print(f"left panel:  b = {result_cap.b:.6f}, outcome = {result_cap.outcome}")
    print(f"right panel: b = {result_esc.b:.6f}, outcome = {result_esc.outcome}")


if __name__ == "__main__":
    plot_nearcritical_side_by_side()