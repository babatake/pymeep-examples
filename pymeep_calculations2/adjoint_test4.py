import numpy as np
from scipy.interpolate import BSpline
import meep as mp
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# === Fixed parameters ===
num_ctrl = 12                      # Number of spline control points
opt_start_idx = 3                 # Index of the first design variable to be optimized
opt_end_idx = 8                   # Index of the last design variable to be optimized
length = 30.0                     # Total waveguide length in microns
x_ctrl = np.linspace(0, length, num_ctrl)  # x-coordinates of control points
base_y_top = 1.5                  # Base height of top boundary
base_y_bot = -1.5                 # Base height of bottom boundary
waveguide_height = 0.4           # Height of the waveguide core
pixel_size = 0.06                 # Spatial resolution of the design mask
resolution = 8                    # Meep resolution (pixels per micron)
wavelength = 1.55                 # Wavelength in microns
fcen = 1 / wavelength             # Center frequency
dpml = 1.0                        # PML thickness
z_thickness = 0.22                # Physical thickness of the waveguide
padding_above = 2.0              # Padding above the waveguide
padding_below = 2.0              # Padding below the waveguide
z_center = -0.5 * (z_thickness + padding_above + padding_below + 2 * dpml) + dpml + padding_below + 0.5 * z_thickness
margin = 0.1                      # Margin between base height and shape

# === Design variables ===
design_params_top = np.zeros(num_ctrl)  # Top spline perturbations
design_params_bot = np.zeros(num_ctrl)  # Bottom spline perturbations
design_step = 1e-3                       # Step size (not used currently)


def make_mask(params_top, params_bot):
    """Generate design mask as a MaterialGrid weight map using B-spline control points."""
    full_top = base_y_top + margin + params_top
    full_bot = base_y_bot - margin - params_bot
    knot_vector = np.linspace(0, 1, num_ctrl + 3 + 1)
    spline_top = BSpline(knot_vector, full_top, 3)
    spline_bot = BSpline(knot_vector, full_bot, 3)

    t_vals = np.linspace(0, 1, 200)
    x_vals = np.linspace(0, length, 200)
    y_vals_top = spline_top(t_vals)
    y_vals_bot = spline_bot(t_vals)

    top = list(zip(x_vals, y_vals_top)) + list(zip(x_vals[::-1], np.full_like(x_vals, base_y_top)[::-1]))
    bot = list(zip(x_vals, np.full_like(x_vals, base_y_bot))) + list(zip(x_vals[::-1], y_vals_bot[::-1]))
    wg1 = Polygon(top).buffer(0)
    wg2 = Polygon(bot).buffer(0)

    merged = unary_union([wg1, wg2])
    xmin, ymin, xmax, ymax = merged.bounds
    xrange = xmax - xmin
    yrange = ymax - ymin

    grid_x = int(np.ceil(xrange / pixel_size))
    grid_y = int(np.ceil(yrange / pixel_size))
    xx = np.linspace(xmin, xmax, grid_x)
    yy = np.linspace(ymin, ymax, grid_y)
    xx_grid, yy_grid = np.meshgrid(xx, yy)

    mask = np.array([
        merged.contains(Point(x, y))
        for x, y in zip(xx_grid.ravel(), yy_grid.ravel())
    ]).reshape((grid_y, grid_x)).astype(float)

    return mask, xmin, xmax, ymin, ymax, xx, yy, x_vals, y_vals_top, y_vals_bot


def run_meep_forward(mask, xmin, xmax, ymin, ymax):
    """Run the forward simulation using the Meep MaterialGrid."""
    si = mp.Medium(index=3.48)
    sio2 = mp.Medium(index=1.44)
    xrange = xmax - xmin
    yrange = ymax - ymin
    sx = xrange + 2 * dpml
    sy = yrange + 2 * dpml
    sz = z_thickness + padding_above + padding_below + 2 * dpml
    cell_size = mp.Vector3(sx, sy, sz)

    grid_size = mp.Vector3(mask.shape[1], mask.shape[0])
    material_grid = mp.MaterialGrid(grid_size=grid_size, medium1=sio2, medium2=si, weights=mask.T)

    geometry = [mp.Block(size=mp.Vector3(xrange, yrange, z_thickness),
                        center=mp.Vector3(0, 0, z_center),
                        material=material_grid)]

    src_x = -0.5 * sx + dpml + 0.1
    sources = [mp.EigenModeSource(src=mp.ContinuousSource(frequency=fcen),
                                  center=mp.Vector3(src_x, 0, z_center),
                                  size=mp.Vector3(0, sy - 2 * dpml, z_thickness),
                                  direction=mp.X,
                                  eig_band=1,
                                  eig_parity=mp.NO_PARITY,
                                  eig_match_freq=True)]

    pml_layers = [mp.PML(dpml)]

    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        default_material=sio2)

    monitor_x = 0.5 * sx - dpml - 0.1
    flux_region = mp.FluxRegion(center=mp.Vector3(monitor_x, 0, z_center),
                                size=mp.Vector3(0, sy - 2 * dpml, z_thickness),
                                direction=mp.X)
    trans_flux = sim.add_flux(fcen, 0, 1, flux_region)

    sim.run(until=200)
    eps_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
    Ey_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ey)

    return mp.get_fluxes(trans_flux)[0], eps_data, Ey_data, sim


def compute_adjoint_field(sim, monitor_pos):
    """Run adjoint simulation and return the adjoint field."""
    adj_source = mp.EigenModeSource(src=mp.ContinuousSource(frequency=fcen),
                                    center=monitor_pos,
                                    size=mp.Vector3(0, sim.cell_size.y - 2 * dpml, z_thickness),
                                    direction=mp.X,
                                    eig_band=1,
                                    eig_parity=mp.NO_PARITY,
                                    eig_match_freq=True,
                                    amplitude=1.0)

    sim.reset_meep()
    sim.change_sources([adj_source])
    sim.run(until=300)
    Ey_adj = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Ey)
    return Ey_adj


# === Optimization loop (Adjoint method) ===
num_iters = 5
lr = 10000
for it in range(num_iters):
    print(f"\n=== Iteration {it+1}/{num_iters} ===")
    mask, xmin, xmax, ymin, ymax, xx, yy, x_vals, y_vals_top, y_vals_bot = make_mask(design_params_top, design_params_bot)
    J0, eps_data, Ey_fwd, sim = run_meep_forward(mask, xmin, xmax, ymin, ymax)
    print(f"Forward J = {J0:.6f}")

    monitor_x = 0.5 * (xmax - xmin) + dpml - 0.1
    Ey_adj = compute_adjoint_field(sim, monitor_pos=mp.Vector3(monitor_x, 0, z_center))

    dJ_deps = 2 * np.real(Ey_fwd[:, :, Ey_fwd.shape[2] // 2] * Ey_adj[:, :, Ey_adj.shape[2] // 2])

    gradient_top = np.zeros_like(design_params_top)
    gradient_bot = np.zeros_like(design_params_bot)

    x_coords = np.linspace(xmin, xmax, eps_data.shape[0])
    y_coords = np.linspace(ymin, ymax, eps_data.shape[1])

    for i in range(opt_start_idx, opt_end_idx):
        xi = x_ctrl[i]
        x_mask = (x_coords >= xi - 5) & (x_coords <= xi + 5)

        yi_top = y_vals_top[np.abs(x_vals - xi).argmin()]
        y_mask_top = (y_coords >= base_y_top) & (y_coords <= yi_top)
        region_top = np.outer(y_mask_top, x_mask)
        gradient_top[i] = np.sum(dJ_deps.T * region_top)

        yi_bot = y_vals_bot[np.abs(x_vals - xi).argmin()]
        y_mask_bot = (y_coords >= yi_bot) & (y_coords <= base_y_bot)
        region_bot = np.outer(y_mask_bot, x_mask)
        gradient_bot[i] = np.sum(dJ_deps.T * region_bot)

    print("∂J/∂x_top:", gradient_top[opt_start_idx:opt_end_idx])
    print("∂J/∂x_bot:", gradient_bot[opt_start_idx:opt_end_idx])

    # Gradient descent update
    design_params_top -= lr * gradient_top
    design_params_bot -= lr * gradient_bot

    print("Updated design_params_top:", design_params_top)
    print("Updated design_params_bot:", design_params_bot)
