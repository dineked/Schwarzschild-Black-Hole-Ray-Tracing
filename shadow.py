import numpy as np
import matplotlib.pyplot as plt

from constants import M
from equatorial_solver import critical_impact_parameter


def make_shadow_grid(screen_size=8.0, resolution=500, ring_width=0.08):
    """
    Construct the idealised Schwarzschild black-hole shadow in
    impact-parameter screen space.

    The screen coordinates are bx and by. The screen-space radius is

        B = sqrt(bx^2 + by^2).

    For a Schwarzschild black hole, the idealised shadow is the set of
    screen directions whose impact-parameter radius satisfies

        B < b_crit,

    where

        b_crit = 3 sqrt(3) M.

    These screen directions correspond to captured null geodesics. Directions
    with B > b_crit correspond to escaping rays.

    Parameters
    ----------
    screen_size:
        Maximum absolute value of bx and by shown on the screen. The plotted
        range is [-screen_size, +screen_size] in both directions.

    resolution:
        Number of grid points along each screen axis.

    ring_width:
        Width of the optional visual band around B = b_crit. This band
        highlights the capture--escape boundary. It is not an emission model.
    """
    bcrit = critical_impact_parameter()

    bx = np.linspace(-screen_size, screen_size, resolution)
    by = np.linspace(-screen_size, screen_size, resolution)

    BX, BY = np.meshgrid(bx, by)
    B = np.sqrt(BX**2 + BY**2)

    # Image values:
    # 0.0 = shadow region: captured rays, B < b_crit
    # 1.0 = escaping region: B > b_crit
    image = np.ones_like(B)
    image[B < bcrit] = 0.0

    # Optional boundary highlight near B = b_crit.
    boundary_band = np.abs(B - bcrit) < ring_width
    image[boundary_band] = 0.6

    return bx, by, image, bcrit


def plot_shadow(screen_size=8.0, resolution=500, ring_width=0.08, save=False,
                filename="fig_shadow_grid.png"):
    """
    Plot the Schwarzschild shadow in impact-parameter screen space.

    This plot is not a spatial trajectory plot. The axes are bx and by,
    which label observer-screen directions through their impact parameters.
    The dashed circle marks

        B = b_crit = 3 sqrt(3) M.

    Inside this circle, rays are classified as captured. Outside it, rays are
    classified as escaping.
    """
    bx, by, image, bcrit = make_shadow_grid(
        screen_size=screen_size,
        resolution=resolution,
        ring_width=ring_width,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(
        image,
        extent=[bx.min(), bx.max(), by.min(), by.max()],
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    boundary = plt.Circle(
        (0, 0),
        bcrit,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        label=rf"$b_{{\rm crit}} = {bcrit:.3f}M$",
    )
    ax.add_patch(boundary)

    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$b_x/M$")
    ax.set_ylabel(r"$b_y/M$")
    ax.set_title("Schwarzschild black-hole shadow in impact-parameter space")
    ax.legend()

    fig.tight_layout()

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def shadow_angular_radius(r_obs):
    """
    Compute the angular radius of the Schwarzschild shadow for a static
    observer outside the photon sphere.

    The observer radius is the Schwarzschild radial coordinate r_obs. The
    angular radius satisfies

        sin(alpha_sh) = b_crit sqrt(1 - 2M/r_obs) / r_obs.

    This is the expression used in the thesis for a static observer outside
    the photon sphere. The returned angle is in radians.
    """
    if r_obs <= 3.0 * M:
        raise ValueError(
            "For this thesis calculation, the observer radius should satisfy "
            "r_obs > 3M."
        )

    bcrit = critical_impact_parameter()
    f_obs = 1.0 - 2.0 * M / r_obs

    argument = bcrit * np.sqrt(f_obs) / r_obs

    # Clip only to avoid round-off problems near the mathematical limit.
    argument = np.clip(argument, -1.0, 1.0)

    return np.arcsin(argument)


def plot_shadow_angular_size(
    r_min=3.1,
    r_max=80.0,
    n=400,
    save=True,
    filename="fig_shadow_angle.png",
):
    """
    Plot the angular radius of the Schwarzschild shadow as a function of
    observer radius.

    The plotted quantity is

        alpha_sh(r_obs),

    where

        sin(alpha_sh) = b_crit sqrt(1 - 2M/r_obs) / r_obs.

    I start the plot outside the photon sphere, with r_obs > 3M, matching
    the assumptions stated in the thesis.
    """
    if r_min <= 3.0 * M:
        raise ValueError("Use r_min > 3M for this angular-size plot.")

    radii = np.linspace(r_min, r_max, n)
    angles = np.array([shadow_angular_radius(r) for r in radii])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(radii, np.degrees(angles))

    ax.set_xlabel(r"Observer radius $r_{\rm obs}/M$")
    ax.set_ylabel(r"Shadow angular radius $\alpha_{\rm sh}$ (degrees)")
    ax.set_title("Angular size of the Schwarzschild shadow")
    ax.grid(True)

    fig.tight_layout()

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    """
    Generate the shadow figures used in the thesis.

    The first plot constructs the idealised capture region in
    impact-parameter space. The second plot shows how the angular radius of
    the same Schwarzschild shadow changes with observer radius.
    """
    plot_shadow(
        screen_size=8.0,
        resolution=600,
        ring_width=0.05,
        save=False,
    )

    plot_shadow_angular_size(
        r_min=3.1,
        r_max=80.0,
        n=400,
        save=True,
    )


if __name__ == "__main__":
    main()