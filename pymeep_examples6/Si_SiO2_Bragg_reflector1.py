#!/usr/bin/env python3
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
lambda0 = 1.55
nH, nL = 2.0, 1.44
num_layers = 20

lambda_min, lambda_max = 1.0, 2.0
nfreq = 400

resolution = 200
dpml = 2.0
dair = 4.0
dsub = 4.0
source_offset = 1.0

t_after = 4000  # ソース終了後に回す時間（必要なら増やす）

# quarter-wave thicknesses
dH = lambda0 / (4 * nH)
dL = lambda0 / (4 * nL)

# freq range
fmin = 1 / lambda_max
fmax = 1 / lambda_min
fcen = 0.5 * (fmin + fmax)
df = fmax - fmin

# layer stack H/L/...
layers = []
for i in range(num_layers):
    if i % 2 == 0:
        layers.append((dH, mp.Medium(index=nH)))  # Si
    else:
        layers.append((dL, mp.Medium(index=nL)))  # SiO2
stack_thickness = sum(th for th, _ in layers)

# ----------------------------
# 1D cell: force axis = z  (official pattern)
# ----------------------------
sz = 2 * dpml + dair + stack_thickness + dsub
cell = mp.Vector3(0, 0, sz)   # ★ここが最重要：1D軸をzに固定

# positions along z
z_front_pml_edge = -0.5 * sz + dpml
z_stack_start = z_front_pml_edge + dair
z_stack_end = z_stack_start + stack_thickness

z_src  = z_front_pml_edge + source_offset
z_refl = z_src + 0.5
z_tran = z_stack_end + 0.5

refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, z_refl))
tran_fr = mp.FluxRegion(center=mp.Vector3(0, 0, z_tran))

COMP = mp.Ex  # ★1D軸=zなので横成分Ex/Eyが使える（EzはNG）

def make_geometry():
    geom = []
    z = z_stack_start
    for th, med in layers:
        geom.append(
            mp.Block(
                material=med,
                center=mp.Vector3(0, 0, z + 0.5*th),
                size=mp.Vector3(0, 0, th)   # ★1Dなのでこれが安全
            )
        )
        z += th
    return geom


def run_empty():
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml, direction=mp.Z)],  # z方向PML
        geometry=[],
        resolution=resolution,
        dimensions=1,
        default_material=mp.Medium(index=1.0)
    )

    sim.sources = [mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df),
        component=COMP,
        center=mp.Vector3(0, 0, z_src)
    )]

    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    sim.run(until_after_sources=t_after)

    freqs = np.array(mp.get_flux_freqs(refl))
    tran_inc = np.array(mp.get_fluxes(tran))
    refl_data = sim.get_flux_data(refl)

    print("[empty] incident power stats:",
          "min=", float(np.min(tran_inc)),
          "max=", float(np.max(tran_inc)),
          "mean=", float(np.mean(tran_inc)))

    sim.reset_meep()
    return freqs, tran_inc, refl_data

def run_structure(freqs, tran_inc, refl_data):
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml, direction=mp.Z)],
        geometry=make_geometry(),
        resolution=resolution,
        dimensions=1,
        default_material=mp.Medium(index=1.0)
    )

    sim.sources = [mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df),
        component=COMP,
        center=mp.Vector3(0, 0, z_src)
    )]

    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    sim.load_minus_flux_data(refl, refl_data)

    sim.run(until_after_sources=t_after)

    refl_only = np.array(mp.get_fluxes(refl))   # signed
    tran_flux = np.array(mp.get_fluxes(tran))

    print("[struct] refl flux stats:",
          "min=", float(np.min(refl_only)),
          "max=", float(np.max(refl_only)),
          "mean=", float(np.mean(refl_only)))
    print("[struct] tran flux stats:",
          "min=", float(np.min(tran_flux)),
          "max=", float(np.max(tran_flux)),
          "mean=", float(np.mean(tran_flux)))

    sim.reset_meep()

    I0 = np.maximum(tran_inc, 1e-30)
    R = np.maximum(0.0, (-refl_only) / I0)
    T = np.maximum(0.0, tran_flux / I0)

    lam = 1 / freqs
    idx = np.argsort(lam)
    return lam[idx], R[idx], T[idx]

# ----------------------------
# run + plot
# ----------------------------
freqs, tran_inc, refl_data = run_empty()
lam, R, T = run_structure(freqs, tran_inc, refl_data)

plt.figure()
plt.plot(lam, R, label="Reflectance R")
plt.plot(lam, T, label="Transmittance T")
plt.plot(lam, np.clip(R + T, 0, 2), label="R+T")
plt.axvline(lambda0, linestyle="--", label="lambda0")
plt.xlabel("Wavelength (um)")
plt.ylabel("Power ratio")
plt.title(f"Si/SiO2 DBR (1D, axis=z): {num_layers} layers, quarter-wave @ {lambda0} um")
plt.grid(True)
plt.legend()
plt.show()

i0 = int(np.argmin(np.abs(lam - lambda0)))
print(f"Nearest to lambda0: lambda={lam[i0]:.4f} um, R={R[i0]:.4f}, T={T[i0]:.4f}, R+T={R[i0]+T[i0]:.4f}")
