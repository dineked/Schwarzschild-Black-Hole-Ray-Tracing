import numpy as np
import matplotlib.pyplot as plt

from constants import M
from equatorial_solver import trace_equatorial_ray, critical_impact_parameter


def flat_space_angle_finite_radius(b, r_start, r_end):
    """
    Compute the flat-space angular change over the same finite radial interval
    used by the numerical Schwarzschild ray.

    The numerical ray is not integrated from infinity to infinity. It starts
    at a finite radius r_start and, if it escapes, stops at a finite radius
    r_end. For that reason, I do not subtract pi directly. I subtract the
    angular change of a straight-line trajectory with the same impact
    parameter b over the same finite radial interval.

    In the limit r_start, r_end -> infinity, this expression approaches pi.
    """
    if b >= min(r_start, r_end):
        raise ValueError(
            f"Impact parameter b={b} must be smaller than both "
            f"r_start={r_start} and r_end={r_end}."
        )

    angle_in = np.arccos(np.clip(b / r_start, -1.0, 1.0))
    angle_out = np.arccos(np.clip(b / r_end, -1.0, 1.0))

    return angle_in + angle_out


def numerical_deflection_angle(result):
    """
    Compute the finite-radius corrected deflection angle for one escaping ray.

    The total angular change of the Schwarzschild ray is

        Delta_phi_GR = |phi_final - phi_initial|.

    Since the ray starts and ends at finite radii, the finite-radius corrected
    deflection angle is

        delta_phi_num = Delta_phi_GR - Delta_phi_flat,

    where Delta_phi_flat is the angular change of a flat-space straight-line
    trajectory over the same radial interval.

    For captured rays, the scattering deflection angle is not defined, because
    the ray never reaches the outward asymptotic branch. In that case I return
    NaN.
    """
    if result.outcome != "escape":
        return np.nan

    delta_phi_gr = abs(result.phi[-1] - result.phi[0])

    r_start = float(result.r[0])
    r_end = float(result.r[-1])

    delta_phi_flat = flat_space_angle_finite_radius(
        b=result.b,
        r_start=r_start,
        r_end=r_end,
    )

    return delta_phi_gr - delta_phi_flat


def weak_field_deflection(b):
    """
    Leading-order weak-field Schwarzschild deflection.

    This approximation is

        delta_phi ≈ 4M / b,

    and is valid when b >> M. It should agree with the numerical ray tracing
    only in the weak-field region.
    """
    return 4.0 * M / b


def second_order_deflection(b):
    """
    Weak-field Schwarzschild deflection through second order.

    The approximation is

        delta_phi ≈ 4M/b + 15 pi M^2/(4 b^2).

    This improves the weak-field comparison at moderately large b, but it is
    still not expected to describe the strong-field growth near b_crit.
    """
    return 4.0 * M / b + (15.0 * np.pi * M**2) / (4.0 * b**2)


def deflection_impact_parameters():
    """
    Construct the impact-parameter values used in the deflection plot.

    Only escaping rays are sampled, so all values satisfy b > b_crit. The
    sampling is dense close to b_crit to show the strong-field growth of the
    deflection angle.
    """
    bcrit = critical_impact_parameter()

    return np.concatenate([
        np.linspace(bcrit + 0.0005, bcrit + 0.01, 25),
        np.linspace(bcrit + 0.01, 6.5, 30),
        np.linspace(6.7, 30.0, 50),
    ])


def compute_deflection_curve(r0=100.0, lam_max=10000.0):
    """
    Trace escaping rays and compute their finite-radius deflection angles.

    The ray tracer itself is implemented in equatorial_solver.py. This
    function only chooses the escaping impact parameters, calls the ray tracer,
    and stores the corrected deflection angles.

    I use r0 = 100M for this calculation because the deflection angle is a
    scattering quantity. Starting farther from the black hole gives a better
    finite-radius approximation to the asymptotic bending angle than the
    compact trajectory plots with r0 = 30M.
    """
    b_values = deflection_impact_parameters()

    numerical = []

    for b in b_values:
        result = trace_equatorial_ray(
            r0=r0,
            b=b,
            lam_max=lam_max,
        )

        numerical.append(numerical_deflection_angle(result))

    return b_values, np.array(numerical)


def plot_deflection_angle(
    r0=100.0,
    lam_max=10000.0,
    save=True,
    filename="deflection_angle.png",
):
    """
    Plot the finite-radius corrected numerical deflection angle.

    The plot compares three quantities:

    1. The numerical Schwarzschild ray-tracing result.
    2. The leading weak-field approximation, delta_phi ≈ 4M/b.
    3. The second-order weak-field approximation.

    The vertical dashed line marks b_crit. Values below b_crit are not plotted,
    because those rays are captured and do not have a scattering deflection
    angle.
    """
    bcrit = critical_impact_parameter()

    b_values, numerical = compute_deflection_curve(
        r0=r0,
        lam_max=lam_max,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        b_values,
        numerical,
        "o",
        markersize=4,
        label="numerical ray tracing",
    )

    ax.plot(
        b_values,
        weak_field_deflection(b_values),
        "--",
        linewidth=1.5,
        label=r"weak-field $\delta\phi \approx 4M/b$",
    )

    ax.plot(
        b_values,
        second_order_deflection(b_values),
        ":",
        linewidth=1.8,
        label=r"second-order weak-field approximation",
    )

    ax.axvline(
        bcrit,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=rf"$b_{{\rm crit}}={bcrit:.4f}$",
    )

    ax.set_xlabel(r"Impact parameter $b/M$")
    ax.set_ylabel(r"Deflection angle $\delta\phi$ (radians)")
    ax.set_title("Deflection angle as a function of impact parameter")
    ax.grid(True)
    ax.legend()

    # Keep the strong-field growth visible without leaving excessive white space.
    finite_values = numerical[np.isfinite(numerical)]
    if finite_values.size > 0:
        finite_max = np.nanmax(finite_values)
        ax.set_ylim(0, max(6.0, 1.05 * finite_max))

    fig.tight_layout()

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_deflection_angle()