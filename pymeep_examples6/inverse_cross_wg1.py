#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

import meep as mp

import jax
import jax.numpy as jnp
import optax


# ----------------------------
# 0) Fixed parameters
# ----------------------------
eps_core = 12.0   # Si
eps_clad = 2.1    # SiO2
mat_core = mp.Medium(epsilon=eps_core)
mat_clad = mp.Medium(epsilon=eps_clad)

resolution = 60        # pixels/um
dpml = 1.0
w = 0.45
design_size = 3.0
sx = 16.0
sy = 16.0
cell = mp.Vector3(sx, sy, 0)

# Wavelength band (um)
lambda_min = 1.50
lambda_max = 1.60
fmin = 1 / lambda_max
fmax = 1 / lambda_min
fcen = 0.5 * (fmin + fmax)
df = (fmax - fmin)

K = 5

# runtime control
decay_by = 1e-9
min_run_time = 80   # used only if your meep supports it (auto-detected)
max_run_time = 600  # used only if your meep supports it (auto-detected)
fallback_until = 500  # safety cap when stop_when_dft_decayed has no min/max args

# Shape params
M = 8
theta_min = w/2
theta_max = 1.10/2

# Smooth/projection
tau = 0.02
rmin = 0.08
eta = 0.5

# Optim
num_iters = 80     # SPSA is expensive (2 forward runs/iter)
lr = 0.03

def beta_schedule(it):
    if it < 25:
        return 2.0
    elif it < 55:
        return 8.0
    else:
        return 32.0


# ----------------------------
# 1) JAX: spline -> rho(x,y)
# ----------------------------
@jax.jit
def _catmull_rom_eval(x, xk, yk):
    i = jnp.clip(jnp.searchsorted(xk, x, side="right") - 1, 0, xk.size - 2)
    y0 = yk[jnp.maximum(i - 1, 0)]
    y1 = yk[i]
    y2 = yk[i + 1]
    y3 = yk[jnp.minimum(i + 2, yk.size - 1)]
    x1 = xk[i]
    x2 = xk[i + 1]
    t = (x - x1) / (x2 - x1 + 1e-12)
    a = 2*y1
    b = -y0 + y2
    c = 2*y0 - 5*y1 + 4*y2 - y3
    d = -y0 + 3*y1 - 3*y2 + y3
    return 0.5 * (a + b*t + c*t*t + d*t*t*t)

def _gaussian_kernel_2d(sigma_pix, radius=3):
    sigma_pix = float(sigma_pix)
    rad = int(max(1, radius * sigma_pix))
    xs = np.arange(-rad, rad + 1)
    ys = np.arange(-rad, rad + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Kk = np.exp(-(X**2 + Y**2) / (2 * sigma_pix**2 + 1e-12))
    Kk /= Kk.sum()
    return jnp.array(Kk, dtype=jnp.float32)

@jax.jit
def _conv2d_same(img, kernel):
    img4 = img[None, None, :, :]
    ker4 = kernel[None, None, :, :]
    out = jax.lax.conv_general_dilated(
        img4, ker4,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW")
    )
    return out[0, 0, :, :]

def make_rho_factory(Nx, Ny, dx):
    x = (jnp.arange(Nx) - (Nx - 1) / 2) * dx
    y = (jnp.arange(Ny) - (Ny - 1) / 2) * dx
    X, Y = jnp.meshgrid(x, y, indexing="xy")

    xk = jnp.linspace(0.0, design_size / 2, M)

    sigma_pix = (rmin / dx)
    gker = _gaussian_kernel_2d(sigma_pix, radius=3)

    @jax.jit
    def make_rho(theta, beta):
        theta_c = jnp.clip(theta, theta_min, theta_max)
        Xabs = jnp.abs(X)
        f = _catmull_rom_eval(jnp.clip(Xabs, 0.0, design_size / 2), xk, theta_c)

        s = jnp.abs(Y) - f
        rho_raw = jax.nn.sigmoid(-s / tau)
        rho_blur = _conv2d_same(rho_raw.astype(jnp.float32), gker)
        rho_proj = jax.nn.sigmoid(beta * (rho_blur - eta))

        arm_mask = (jnp.abs(X) <= w/2) | (jnp.abs(Y) <= w/2)
        rho = jnp.where(arm_mask, 1.0, rho_proj)
        return jnp.clip(rho, 0.0, 1.0)

    return make_rho


# ----------------------------
# Helpers: flux retrieval + robust stop condition
# ----------------------------
def _get_fluxes(sim, flux_obj):
    """Compatibility: mp.get_fluxes(obj) vs sim.get_fluxes(obj)."""
    if hasattr(mp, "get_fluxes"):
        return mp.get_fluxes(flux_obj)
    return sim.get_fluxes(flux_obj)

def _run_until_dft_decayed(sim, decay, dft_obj):
    """
    Compatibility wrapper for stop_when_dft_decayed.
    Tries signatures:
      (decay, dft_obj, min_time, max_time)
      (decay, dft_obj)
    and falls back to fixed until=... if needed.
    """
    stop = None
    try:
        # Newer signature (some builds)
        stop = mp.stop_when_dft_decayed(decay, dft_obj, min_run_time, max_run_time)
        sim.run(until_after_sources=stop)
        return
    except TypeError:
        pass

    try:
        # Older signature
        stop = mp.stop_when_dft_decayed(decay, dft_obj)
        # Some versions allow adding a safety cap via until=...
        try:
            sim.run(until_after_sources=stop, until=fallback_until)
        except TypeError:
            sim.run(until_after_sources=stop)
        return
    except TypeError:
        pass

    # Oldest fallback: just run a fixed time
    sim.run(until=fallback_until)


# ----------------------------
# 2) Meep forward: rho -> normalized flux metrics -> loss
# ----------------------------
def run_forward_flux(rho_2d: np.ndarray):
    """
    rho_2d: (Ny,Nx) in [0,1]
    returns: loss(float), (Ts, Tx, R) each shape (K,)
    """
    Ny, Nx = rho_2d.shape[0], rho_2d.shape[1]

    matgrid = mp.MaterialGrid(
        mp.Vector3(Nx, Ny),
        mat_clad,
        mat_core,
        grid_type="U_MEAN",
        weights=rho_2d
    )

    arm_x = mp.Block(size=mp.Vector3(sx, w, mp.inf), center=mp.Vector3(0, 0), material=mat_core)
    arm_y = mp.Block(size=mp.Vector3(w, sy, mp.inf), center=mp.Vector3(0, 0), material=mat_core)
    design_block = mp.Block(size=mp.Vector3(design_size, design_size, mp.inf), center=mp.Vector3(0, 0), material=matgrid)
    geometry = [arm_x, arm_y, design_block]

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            center=mp.Vector3(-sx/2 + dpml + 1.0, 0, 0),
            size=mp.Vector3(0, 2.0*w, 0),
            direction=mp.X,
            eig_band=1,
            eig_match_freq=True,
            component=mp.Ez,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        default_material=mat_clad,
        dimensions=2
    )

    mon_w = 2.0*w

    flux_in = sim.add_flux(
        fcen, df, K,
        mp.FluxRegion(center=mp.Vector3(-sx/2 + dpml + 2.2, 0, 0), size=mp.Vector3(0, mon_w, 0))
    )
    flux_right = sim.add_flux(
        fcen, df, K,
        mp.FluxRegion(center=mp.Vector3(+sx/2 - dpml - 1.0, 0, 0), size=mp.Vector3(0, mon_w, 0))
    )
    flux_top = sim.add_flux(
        fcen, df, K,
        mp.FluxRegion(center=mp.Vector3(0, +sy/2 - dpml - 1.0, 0), size=mp.Vector3(mon_w, 0, 0))
    )
    flux_left = sim.add_flux(
        fcen, df, K,
        mp.FluxRegion(center=mp.Vector3(-sx/2 + dpml + 0.6, 0, 0), size=mp.Vector3(0, mon_w, 0))
    )

    # robust run (handles your old stop_when_dft_decayed signature)
    _run_until_dft_decayed(sim, decay_by, flux_right)

    Pin = np.array(_get_fluxes(sim, flux_in), dtype=float)
    Pr  = np.array(_get_fluxes(sim, flux_right), dtype=float)
    Pt  = np.array(_get_fluxes(sim, flux_top), dtype=float)
    Pl  = np.array(_get_fluxes(sim, flux_left), dtype=float)

    Pin = np.maximum(Pin, 1e-12)
    Ts = Pr / Pin
    Tx = Pt / Pin
    R = np.maximum(-Pl / Pin, 0.0)  # reflection tends to show as negative flux

    ws, wx, wr = 1.0, 1.0, 0.5
    eps = 1e-9
    loss = float(np.mean(-ws * np.log(Ts + eps) + wx * Tx + wr * R))

    return loss, Ts, Tx, R


# ----------------------------
# 3) SPSA optimization loop (2 forward sims per iter)
# ----------------------------
def main():
    dx = 1.0 / resolution
    Nx = int(round(design_size * resolution))
    Ny = Nx
    print(f"Design grid: {Nx} x {Ny}, dx={dx:.5f} um")

    make_rho = make_rho_factory(Nx, Ny, dx)

    theta0 = np.ones((M,), dtype=np.float32) * (w/2)
    theta0[0] = min(theta_max, (0.65/2))
    theta0[1] = min(theta_max, (0.60/2))
    theta = jnp.array(theta0)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta)

    # SPSA schedules
    a0 = 0.08
    c0 = 0.03
    A = 10.0
    alpha = 0.602
    gamma = 0.101

    best = {"loss": 1e99, "theta": None}

    for it in range(num_iters):
        beta = beta_schedule(it)

        ak = a0 / ((it + 1 + A) ** alpha)
        ck = c0 / ((it + 1) ** gamma)

        delta = np.random.choice([-1.0, 1.0], size=(M,)).astype(np.float32)
        delta_j = jnp.array(delta)

        theta_p = jnp.clip(theta + ck * delta_j, theta_min, theta_max)
        theta_m = jnp.clip(theta - ck * delta_j, theta_min, theta_max)

        t0 = time.time()
        rho_p = np.array(make_rho(theta_p, beta), dtype=np.float64)
        fp, _, _, _ = run_forward_flux(rho_p)

        rho_m = np.array(make_rho(theta_m, beta), dtype=np.float64)
        fm, _, _, _ = run_forward_flux(rho_m)
        t1 = time.time()

        # SPSA gradient estimate (per-parameter)
        ghat = (fp - fm) / (2.0 * ck) * delta
        g_theta = jnp.array(ghat, dtype=jnp.float32)

        updates, opt_state = optimizer.update(g_theta, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        theta = jnp.clip(theta, theta_min, theta_max)

        f_est = 0.5 * (fp + fm)
        if f_est < best["loss"]:
            best["loss"] = f_est
            best["theta"] = np.array(theta)

        if it % 5 == 0 or it == num_iters - 1:
            print(f"it={it:04d} beta={beta:5.1f} loss~={f_est:.6e} ak={ak:.3e} ck={ck:.3e} dt={t1-t0:.1f}s "
                  f"theta[min,max]=({float(theta.min()):.4f},{float(theta.max()):.4f})")

    theta_best = jnp.array(best["theta"])
    rho_best = np.array(make_rho(theta_best, beta_schedule(num_iters-1)))
    np.save("theta_best.npy", np.array(theta_best))
    np.save("rho_best.npy", rho_best)
    print("Saved: theta_best.npy, rho_best.npy  (best loss~={:.6e})".format(best["loss"]))


if __name__ == "__main__":
    main()
