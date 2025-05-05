import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def main():
    # === パラメータ ===
    n = 3.4           # 基本屈折率
    w = 0.5           # 導波路幅 [μm]
    h = 0.22          # 導波路高さ [μm]
    dpml = 2.0        # PML厚さ [μm]
    sx = 10.0 + 2*dpml
    sy = 2.0 + 2*dpml
    sz = 2.0 + 2*dpml
    cell = mp.Vector3(sx, sy, sz)

    # === 非線形媒質定義 ===
    class Chi2Material(mp.Medium):
        def __init__(self, index):
            super().__init__(index=index)
            self.chi2 = 1.0  # 相対値（物理単位ではない）

    nonlinear_medium = Chi2Material(index=n)

    # === ジオメトリ ===
    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, w, h),
            center=mp.Vector3(),
            material=nonlinear_medium
        )
    ]

    # === 光源設定 ===
    fcen = 0.64
    df = 0.1
    nfreq = 100

    source_pos = mp.Vector3(-0.5 * sx + dpml + 1, 0, 0)
    src = [mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=source_pos,
        size=mp.Vector3(0, w, h),
        direction=mp.X,
        eig_band=1,
        eig_parity=mp.ODD_Z,
        eig_match_freq=True
    )]

    # === モニター位置 ===
    input_pt = source_pos + mp.Vector3(1.0, 0, 0)  # 光源直後
    trans_pt = mp.Vector3(0.5 * sx - dpml - 1, 0, 0)

    # === シミュレーション設定 ===
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=src,
        resolution=20,
        boundary_layers=[mp.PML(dpml)],
        dimensions=3,
    )

    # === フラックスモニター設定 ===
    input_flux = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=input_pt, size=mp.Vector3(0, w, h)))
    trans_flux = sim.add_flux(fcen, df, nfreq,
        mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, w, h)))
    shg_flux = sim.add_flux(2*fcen, df, nfreq,
        mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, w, h)))

    # === 実行 ===
    sim.use_output_directory()
    sim.run(
        mp.at_beginning(mp.output_epsilon),
        until_after_sources=500
    )

    # === 結果取得 ===
    freqs = mp.get_flux_freqs(trans_flux)
    wl = 1 / np.array(freqs)

    input_data = mp.get_fluxes(input_flux)
    trans_data = mp.get_fluxes(trans_flux)
    shg_data = mp.get_fluxes(shg_flux)

    # === 効率の計算 ===
    trans_ratio = np.array(trans_data) / np.array(input_data)
    shg_efficiency = np.array(shg_data) / np.array(input_data)

    # === プロット ===
    plt.figure()
    plt.plot(wl, trans_ratio, label="Transmission (Fundamental)")
    plt.plot(wl / 2, shg_efficiency, label="SHG Efficiency")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Relative Output / Input")
    plt.title("Nonlinear Transmission and SHG Efficiency")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

    # === Ez電場分布可視化 ===
    ez_data = sim.get_array(center=mp.Vector3(z=0), size=mp.Vector3(sx, sy, 0), component=mp.Ez)
    extent = [-0.5 * sx, 0.5 * sx, -0.5 * sy, 0.5 * sy]

    plt.figure()
    plt.imshow(np.rot90(ez_data), interpolation='spline36', cmap='RdBu', extent=extent)
    plt.colorbar(label="Ez")
    plt.title("Ez Field Distribution (z=0)")
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.tight_layout()
    plt.show()
    
    return wl, shg_efficiency

if __name__ == "__main__":
    wl, shg_efficiency = main()
    target_lambda = 1.55
    idx = (np.abs(wl - target_lambda)).argmin()
    print(f"λ = {wl[idx]:.4f} μm における SHG 効率: {shg_efficiency[idx]:.6e}")

