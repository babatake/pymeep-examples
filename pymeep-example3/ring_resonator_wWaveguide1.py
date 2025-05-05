import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def main():
    # === Parameters ===
    n = 3.4
    w = 0.5
    h = 0.22
    r = 10.0
    gap = 0.05
    dpml = 2.0
    pad = 2.0
    sx = 2 * (r + w + gap + pad + dpml)
    sy = 2 * (r + w + gap + pad + dpml)
    sz = h + 2 * dpml + 1.0  # Ensure enough space in the z-direction
    cell = mp.Vector3(sx, sy, sz)

    # === Geometry setup (all z-centers are set to 0) ===
    bus_y = 0
    ring_center_y = w + gap + r
    drop_y = 2 * r + 2 * gap + 2 * w

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, w, h),
            center=mp.Vector3(0, bus_y, 0),
            material=mp.Medium(index=n)
        ),
        mp.Block(
            size=mp.Vector3(mp.inf, w, h),
            center=mp.Vector3(0, drop_y, 0),
            material=mp.Medium(index=n)
        ),
        mp.Cylinder(
            radius=r + w / 2, height=h,
            center=mp.Vector3(0, ring_center_y, 0),
            material=mp.Medium(index=n)
        ),
        mp.Cylinder(
            radius=r - w / 2, height=h,
            center=mp.Vector3(0, ring_center_y, 0),
            material=mp.Medium(index=1.0)
        ),
    ]

    # === Source settings ===
    fcen = 0.64
    df = 0.1
    nfreq = 50
    src = [mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(-0.5 * sx + dpml + 1, 0, 0),
        size=mp.Vector3(0, w, h),
        direction=mp.X,
        eig_band=1,
        eig_parity=mp.ODD_Z,
    )]

    # === Monitor locations (centered at z=0) ===
    trans_pt = mp.Vector3(0.5 * sx - dpml - 1, 0, 0)
    drop_pt = mp.Vector3(0.5 * sx - dpml - 1, ring_center_y, 0)

    # === Simulation definition ===
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=src,
        resolution=40,
        boundary_layers=[mp.PML(dpml)],
        dimensions=3,
    )

    # === Flux monitors ===
    trans_flux = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, w, h)))
    drop_flux = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=drop_pt, size=mp.Vector3(0, w, h)))

    # === Run simulation ===
    sim.use_output_directory()
    sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.at_every(1 / fcen / 20, mp.output_efield_z),
        until_after_sources=1000
    )

    # === Retrieve spectrum ===
    flux_freqs = mp.get_flux_freqs(trans_flux)
    trans_data = mp.get_fluxes(trans_flux)
    drop_data = mp.get_fluxes(drop_flux)
    wavelengths = [1 / f for f in flux_freqs]

    # === Plot transmission/drop spectrum ===
    plt.figure()
    plt.plot(wavelengths, trans_data, label="Transmission")
    plt.plot(wavelengths, drop_data, label="Drop")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Flux")
    plt.title("Ring Resonator Spectrum")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

    # === Ez field visualization (2D slice at z = 0) ===
    ez_center = mp.Vector3(0, ring_center_y, 0)
    ez_size = mp.Vector3(sx, sy, 0)  # 2D slice (thickness 0 in z)
    ez_data = sim.get_array(center=ez_center, size=ez_size, component=mp.Ez)

    extent = [-0.5 * sx, 0.5 * sx,
              ring_center_y - 0.5 * sy, ring_center_y + 0.5 * sy]

    plt.figure()
    plt.imshow(np.rot90(ez_data), interpolation='spline36', cmap='RdBu',
               extent=extent)
    plt.colorbar(label="Ez")
    plt.title("Ez Field Distribution (z = 0 plane)")
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
