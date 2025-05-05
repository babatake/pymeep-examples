import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def main():
    # === パラメータ ===
    n = 3.4
    w = 0.5
    h = 0.22
    r = 10.0
    gap = 0.05
    dpml = 2.0
    pad = 2.0
    sx = 2 * (r + w + gap + pad + dpml)
    sy = 2 * (r + w + gap + pad + dpml)
    sz = h + 2 * dpml + 1.0  # z方向にも十分な空間を確保
    cell = mp.Vector3(sx, sy, sz)

    # === ジオメトリ設定（z中心はすべて0にする） ===
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

    # === 光源設定 ===
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

    # === モニター位置（z=0中心） ===
    trans_pt = mp.Vector3(0.5 * sx - dpml - 1, 0, 0)
    drop_pt = mp.Vector3(0.5 * sx - dpml - 1, ring_center_y, 0)

    # === シミュレーション定義 ===
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=src,
        resolution=40,
        boundary_layers=[mp.PML(dpml)],
        dimensions=3,
    )

    # === フラックスモニター ===
    trans_flux = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, w, h)))
    drop_flux = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=drop_pt, size=mp.Vector3(0, w, h)))

    # === 実行 ===
    sim.use_output_directory()
    sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.at_every(1 / fcen / 20, mp.output_efield_z),
        until_after_sources=1000
    )

    # === スペクトル取得 ===
    flux_freqs = mp.get_flux_freqs(trans_flux)
    trans_data = mp.get_fluxes(trans_flux)
    drop_data = mp.get_fluxes(drop_flux)
    wavelengths = [1 / f for f in flux_freqs]

    # === スペクトルプロット ===
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

    # === Ez場 可視化（z=0平面で2D断面） ===
    ez_center = mp.Vector3(0, ring_center_y, 0)
    ez_size = mp.Vector3(sx, sy, 0)  # 2Dスライス（z方向厚み0）
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
