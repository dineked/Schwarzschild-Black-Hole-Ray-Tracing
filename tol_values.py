import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import equatorial_solver as eq


def total_angular_displacement(result):
    """
    Compute the total angular displacement along the stored photon trajectory.

    The stored ray contains phi(lambda), so I define

        Delta phi = |phi_final - phi_initial|.

    This quantity is used for both captured and escaping rays.
    """
    return abs(result.phi[-1] - result.phi[0])


def approximate_winding_number(result):
    """
    Compute the approximate winding diagnostic used in the thesis.

    I define

        N ≈ Delta phi / (2 pi).

    This is not an exact number of circular photon orbits. It measures how
    much angular displacement the numerical ray accumulates before it reaches
    its terminal event.
    """
    return total_angular_displacement(result) / (2.0 * np.pi)


def flat_space_angle_finite_radius(b, r_start, r_end):
    """
    Compute the flat-space angular change over the same finite radial interval
    used by the numerical ray.

    This is the finite-radius reference used for the corrected deflection
    angle. In the limit r_start, r_end -> infinity, this tends to pi.
    """
    if b >= min(r_start, r_end):
        return np.nan

    return (
        np.arccos(np.clip(b / r_start, -1.0, 1.0))
        +
        np.arccos(np.clip(b / r_end, -1.0, 1.0))
    )


def finite_radius_deflection_angle(result):
    """
    Compute the finite-radius corrected deflection angle for escaping rays.

    For an escaping ray,

        delta_phi_num = Delta_phi_GR - Delta_phi_flat.

    Captured rays do not have a scattering deflection angle, because they do
    not reach an outward asymptotic branch. For those rays, I return NaN.
    """
    if result.outcome != "escape":
        return np.nan

    b = result.b
    r_start = float(result.r[0])
    r_end = float(result.r[-1])

    delta_phi_flat = flat_space_angle_finite_radius(
        b=b,
        r_start=r_start,
        r_end=r_end,
    )

    if not np.isfinite(delta_phi_flat):
        return np.nan

    delta_phi_gr = total_angular_displacement(result)

    return delta_phi_gr - delta_phi_flat


def save_solver_settings():
    """
    Store the current numerical settings from equatorial_solver.py.

    The tolerance study temporarily changes these values, runs one ray, and
    then restores the original values.
    """
    return {
        "RTOL": eq.RTOL,
        "ATOL": eq.ATOL,
        "MAX_STEP": eq.MAX_STEP,
        "RESTART_TOL": eq.RESTART_TOL,
        "TURN_TOL": eq.TURN_TOL,
    }


def restore_solver_settings(settings):
    """
    Restore numerical settings in equatorial_solver.py after a test run.
    """
    eq.RTOL = settings["RTOL"]
    eq.ATOL = settings["ATOL"]
    eq.MAX_STEP = settings["MAX_STEP"]
    eq.RESTART_TOL = settings["RESTART_TOL"]
    eq.TURN_TOL = settings["TURN_TOL"]


def apply_solver_settings(*, rtol, atol, max_step, restart_tol, turn_tol):
    """
    Apply one temporary set of numerical parameters to equatorial_solver.py.
    """
    eq.RTOL = rtol
    eq.ATOL = atol
    eq.MAX_STEP = max_step
    eq.RESTART_TOL = restart_tol
    eq.TURN_TOL = turn_tol


def run_ray_with_settings(
    *,
    b,
    r0,
    lam_max,
    rtol,
    atol,
    max_step,
    restart_tol,
    turn_tol,
):
    """
    Run one near-critical ray with temporary numerical settings.

    This function is used for the tolerance tests in the thesis. It changes
    the solver tolerances, traces one ray, and restores the original solver
    settings afterwards.

    The ray tracer itself still uses the same physical stopping rules:

        capture: r = 2M + HORIZON_EPS,
        escape:  r = R_ESCAPE on the outward branch.
    """
    original_settings = save_solver_settings()

    try:
        apply_solver_settings(
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            restart_tol=restart_tol,
            turn_tol=turn_tol,
        )

        result = eq.trace_equatorial_ray(
            r0=r0,
            b=b,
            lam_max=lam_max,
        )

    finally:
        restore_solver_settings(original_settings)

    return result


def summarize_result(result, settings_name, rtol, atol, max_step, restart_tol, turn_tol):
    """
    Convert one traced ray into one row for the tolerance-study table.
    """
    delta_phi = total_angular_displacement(result)

    return {
        "setting": settings_name,
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "restart_tol": restart_tol,
        "turn_tol": turn_tol,
        "b": result.b,
        "outcome": result.outcome,
        "turning_radius": result.turning_point_r,
        "min_r": float(np.min(result.r)),
        "final_r": float(result.r[-1]),
        "Delta_phi": delta_phi,
        "N_approx": approximate_winding_number(result),
        "deflection_angle": finite_radius_deflection_angle(result),
        "n_steps": len(result.lam),
        "lambda_final": float(result.lam[-1]),
    }


def solver_tolerance_study(
    b=5.1962,
    r0=30.0,
    lam_max=5000.0,
    max_step=0.005,
    restart_tol=1e-4,
    turn_tol=1e-6,
):
    """
    Study sensitivity to the adaptive solve_ivp tolerances.

    This test changes only rtol and atol. The branch-switch tolerance is kept
    fixed at the thesis value

        RESTART_TOL = 1e-4.

    The purpose is to check whether the capture--escape classification and
    angular diagnostics change when the solver tolerances are tightened.
    """
    settings = [
        ("loose",    1e-8,  1e-10),
        ("standard", 1e-10, 1e-12),
        ("strict",   1e-12, 1e-14),
    ]

    rows = []
    results = []

    for name, rtol, atol in settings:
        print(f"Running solver tolerance setting: {name}, rtol={rtol}, atol={atol}")

        result = run_ray_with_settings(
            b=b,
            r0=r0,
            lam_max=lam_max,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            restart_tol=restart_tol,
            turn_tol=turn_tol,
        )

        rows.append(
            summarize_result(
                result=result,
                settings_name=name,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                restart_tol=restart_tol,
                turn_tol=turn_tol,
            )
        )

        results.append((name, result))

    df = pd.DataFrame(rows)
    df.to_csv("tolerance_solver_table.csv", index=False)

    return df, results


def restart_tolerance_study(
    b=5.1962,
    r0=30.0,
    lam_max=5000.0,
    rtol=1e-10,
    atol=1e-12,
    max_step=0.005,
    turn_tol=1e-6,
):
    """
    Study sensitivity to the branch-switch tolerance.

    For escaping rays, the inward branch stops at

        r_stop = r_turn + RESTART_TOL,

    then the outward branch starts from the same radius neighbourhood.

    This test changes RESTART_TOL while keeping the adaptive solve_ivp
    tolerances fixed. It checks how close the code can move toward the
    analytical turning radius before round-off near R(r; b) = 0 becomes
    relevant.
    """
    settings = [
        ("restart_1e-3", 1e-3),
        ("restart_1e-4", 1e-4),
        ("restart_1e-5", 1e-5),
        ("restart_1e-6", 1e-6),
    ]

    rows = []
    results = []

    for name, restart_tol in settings:
        print(f"Running restart tolerance setting: {name}, RESTART_TOL={restart_tol}")

        result = run_ray_with_settings(
            b=b,
            r0=r0,
            lam_max=lam_max,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            restart_tol=restart_tol,
            turn_tol=turn_tol,
        )

        rows.append(
            summarize_result(
                result=result,
                settings_name=name,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                restart_tol=restart_tol,
                turn_tol=turn_tol,
            )
        )

        results.append((name, result))

    df = pd.DataFrame(rows)
    df.to_csv("tolerance_restart_table.csv", index=False)

    return df, results


def plot_overlay(
    results,
    *,
    filename,
    title,
    xlim=(-4.5, 8.0),
    ylim=(-5.0, 6.0),
):
    """
    Plot several near-critical trajectories on the same axes.

    This plot checks whether changing numerical tolerances visibly changes
    the trajectory. The event horizon and photon sphere are included as
    reference curves.
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    horizon = plt.Circle(
        (0, 0),
        eq.RS,
        color="black",
        zorder=10,
        label=r"event horizon $r=2M$",
    )
    ax.add_patch(horizon)

    photon_sphere = plt.Circle(
        (0, 0),
        3.0 * eq.M,
        fill=False,
        linestyle=":",
        linewidth=2.0,
        color="gray",
        zorder=9,
        label=r"photon sphere $r=3M$",
    )
    ax.add_patch(photon_sphere)

    for name, result in results:
        x, y = eq.to_cartesian(result)

        ax.plot(
            x,
            y,
            linewidth=1.6,
            label=name,
            alpha=0.85,
        )

    ax.set_aspect("equal", "box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def format_table_for_latex(df, filename, caption, label):
    """
    Save a compact LaTeX table for thesis use.

    The selected columns are the ones relevant to the numerical validation:
    tolerances, classification, turning radius, angular displacement,
    deflection angle, and number of solver steps.
    """
    selected = df[
        [
            "setting",
            "rtol",
            "atol",
            "restart_tol",
            "outcome",
            "turning_radius",
            "Delta_phi",
            "deflection_angle",
            "n_steps",
        ]
    ].copy()

    selected["turning_radius"] = pd.to_numeric(
        selected["turning_radius"],
        errors="coerce",
    ).round(8)

    selected["Delta_phi"] = pd.to_numeric(
        selected["Delta_phi"],
        errors="coerce",
    ).round(8)

    selected["deflection_angle"] = pd.to_numeric(
        selected["deflection_angle"],
        errors="coerce",
    ).round(8)

    latex = selected.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
    )

    with open(filename, "w") as f:
        f.write(latex)


def main():
    """
    Run the tolerance checks used to validate the near-critical ray tracing.

    The test ray is

        b = 5.1962,

    which lies slightly above b_crit. It is close enough to the capture--escape
    boundary to test the branch-switch behaviour, but it is still classified
    as escaping for the thesis settings.
    """
    b_test = 5.1962
    bcrit = eq.critical_impact_parameter()

    print("Tolerance study for near-critical escaping ray")
    print("----------------------------------------------")
    print(f"b_test = {b_test:.12f}")
    print(f"b_crit = {bcrit:.12f}")
    print()

    solver_df, solver_results = solver_tolerance_study(b=b_test)
    restart_df, restart_results = restart_tolerance_study(b=b_test)

    print("\nSolver tolerance study:")
    print(solver_df)

    print("\nRestart tolerance study:")
    print(restart_df)

    format_table_for_latex(
        solver_df,
        filename="tolerance_solver_table.tex",
        caption=(
            "Sensitivity of the near-critical escaping trajectory to "
            "adaptive solver tolerances."
        ),
        label="tab:solver_tolerance_study",
    )

    format_table_for_latex(
        restart_df,
        filename="tolerance_restart_table.tex",
        caption=(
            "Sensitivity of the near-critical escaping trajectory to the "
            "branch-switch tolerance."
        ),
        label="tab:restart_tolerance_study",
    )

    plot_overlay(
        solver_results,
        filename="fig_tolerance_solver_overlay.png",
        title=r"Trajectory sensitivity to solver tolerances",
    )

    plot_overlay(
        restart_results,
        filename="fig_tolerance_restart_overlay.png",
        title=r"Trajectory sensitivity to branch-switch tolerance",
    )


if __name__ == "__main__":
    main()