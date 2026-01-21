#!/usr/bin/env python3
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import math

# ----------------------------
# Parameters
# ----------------------------
lambda0 = 1.55
nH, nL = 2.0, 1.44
num_layers = 20

lambda_min, lambda_max = 1.0, 2.0
nfreq = 400

resolution = 60          # まずは軽く。形がOKなら 100〜200へ
dpml = 2.0
dair = 4.0
dsub = 4.0
source_offset = 1.0

theta_deg = 20.0         # 入射角（空気側、法線=yに対して）
t_after = 3000           # ソース終了後に回す時間（必要なら増やす）

# ----------------------------
# Derived
# ----------------------------
dH = lambda0 / (4 * nH)
dL = lambda0 / (4 * nL)

fmin = 1 / lambda_max
fmax = 1 / lambda_min
fcen = 0.5*(fmin+fmax)
df = fmax - fmin

layers = []
for i in range(num_layers):
    if i % 2 == 0:
        layers.append((dH, mp.Medium(index=nH)))
    else:
        layers.append((dL, mp.Medium(index=nL)))
stack_thickness = sum(th for th, _ in layers)

# ----------------------------
# 2D cell is x–y (z=0)
# ----------------------------
Lx = 1.0
sy = 2*dpml + dair + stack_thickness + dsub
cell = mp.Vector3(Lx, sy, 0)

# positions along y (stacking/prop direction)
y_front_pml_edge = -0.5*sy + dpml
y_stack_start = y_front_pml_edge + dair
y_stack_end = y_stack_start + stack_thickness

y_src  = y_front_pml_edge + source_offset
y_refl = y_src + 0.5
y_tran = y_stack_end + 0.5

refl_fr = mp.FluxRegion(center=mp.Vector3(0, y_refl, 0), size=mp.Vector3(Lx, 0, 0))
tran_fr = mp.FluxRegion(center=mp.Vector3(0, y_tran, 0), size=mp.Vector3(Lx, 0, 0))

# PML only along y
pml_layers = [mp.PML(dpml, direction=mp.Y)]

# polarization: TE (out-of-plane E) for x–y 2D
COMP = mp.Ez

# angle via Bloch k_point (along x)
theta = math.radians(theta_deg)
kx = fmin * math.sin(theta)        # broadband representative (official-style)
k_point = mp.Vector3(kx, 0, 0)

# ----------------------------
# Geometry
# ----------------------------
def make_geometry():
    geom = []
    y = y_stack_start
    for th, med in layers:
        geom.append(mp.Block(
            material=med,
            center=mp.Vector3(0, y + 0.5*th, 0),
            size=mp.Vector3(mp.inf, th, mp.inf)  # infinite in x
        ))
        y += th
    return geom

def make_source():
    return [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                      component=COMP,
                      center=mp.Vector3(0, y_src, 0),
                      size=mp.Vector3(Lx, 0, 0))]

# ----------------------------
# Runs
# ----------------------------
def run_empty():
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[],
        sources=make_source(),
        k_point=k_point,
        dimensions=2,
        resolution=resolution,
        default_material=mp.Medium(index=1.0)
    )

    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    sim.run(until_after_sources=t_after)

    freqs = np.array(mp.get_flux_freqs(refl))
    tran_inc = np.array(mp.get_fluxes(tran))
    refl_data = sim.get_flux_data(refl)

    print("[empty] tran_inc stats:", float(np.min(tran_inc)), float(np.max(tran_inc)), float(np.mean(tran_inc)))

    sim.reset_meep()
    return freqs, tran_inc, refl_data

def run_structure(freqs, tran_inc, refl_data):
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=make_geometry(),
        sources=make_source(),
        k_point=k_point,
        dimensions=2,
        resolution=resolution,
        default_material=mp.Medium(index=1.0)
    )

    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    sim.load_minus_flux_data(refl, refl_data)

    sim.run(until_after_sources=t_after)

    refl_only = np.array(mp.get_fluxes(refl))   # signed
    tran_flux = np.array(mp.get_fluxes(tran))

    print("[struct] refl mean, tran mean:", float(np.mean(refl_only)), float(np.mean(tran_flux)))

    sim.reset_meep()

    I0 = np.maximum(tran_inc, 1e-30)
    R = np.maximum(0.0, (-refl_only) / I0)
    T = np.maximum(0.0, tran_flux / I0)

    lam = 1 / freqs
    idx = np.argsort(lam)
    return lam[idx], R[idx], T[idx]

# ----------------------------
# Run + plot
# ----------------------------
freqs, tran_inc, refl_data = run_empty()
lam, R, T = run_structure(freqs, tran_inc, refl_data)

plt.figure()
plt.plot(lam, R, label=f"R (TE: Ez), theta={theta_deg}°")
plt.plot(lam, T, label="T")
plt.plot(lam, np.clip(R+T, 0, 2), label="R+T")
plt.axvline(lambda0, linestyle="--", label="lambda0")
plt.xlabel("Wavelength (um)")
plt.ylabel("Power ratio")
plt.title(f"DBR oblique incidence (2D x–y): nH={nH}, nL={nL}, layers={num_layers}")
plt.grid(True)
plt.legend()
plt.show()
