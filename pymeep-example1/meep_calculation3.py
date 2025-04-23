import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import gdstk
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# --- Waveguide positioning (can be changed easily) ---
waveguide_x_start = 0.0  # μm
waveguide_x_end = 10.0   # μm
x_shift = -waveguide_x_start

# --- GDS parameters ---
gds_file = "straight_waveguide2.gds"
target_layer = (1, 0)

# --- Simulation parameters ---
pixel_size = 0.04
resolution = 10
wavelength = 1.55
fcen = 1 / wavelength
df = 0.0
dpml = 1.0
z_thickness = 0.22
padding_above = 2.0
padding_below = 2.0

# --- Materials ---
si = mp.Medium(index=3.48)
sio2 = mp.Medium(index=1.44)

# --- Read and shift GDS polygons ---
lib = gdstk.read_gds(gds_file)
cell = lib.top_level()[0]
polygons = []

for poly in cell.polygons:
    if (poly.layer, poly.datatype) == target_layer:
        shifted = [(x + x_shift, y) for x, y in poly.points]
        polygons.append(Polygon(shifted))
merged = unary_union(polygons)

# --- Bitmap creation ---
bbox = merged.bounds
xrange = bbox[0], bbox[2]
yrange = bbox[1], bbox[3]
grid_x = int((xrange[1] - xrange[0]) / pixel_size)
grid_y = int((yrange[1] - yrange[0]) / pixel_size)

xx, yy = np.meshgrid(
    np.linspace(xrange[0], xrange[1], grid_x),
    np.linspace(yrange[0], yrange[1], grid_y)
)

bitmap = np.array([
    merged.contains(Point(x, y)) for x, y in zip(xx.ravel(), yy.ravel())
]).astype(float).reshape((grid_y, grid_x))

# --- MaterialGrid ---
grid_size = mp.Vector3(bitmap.shape[1], bitmap.shape[0])
material_grid = mp.MaterialGrid(grid_size, sio2, si, weights=bitmap.T)

# --- Cell & geometry ---
sx = (xrange[1] - xrange[0]) + 2 * dpml
sy = (yrange[1] - yrange[0]) + 2 * dpml
sz = z_thickness + padding_above + padding_below + 2 * dpml
cell_size = mp.Vector3(sx, sy, sz)

x_center = -0.5 * sx + dpml + 0.5 * (xrange[1] - xrange[0])
y_center = -0.5 * sy + dpml + 0.5 * (yrange[1] - yrange[0])
z_center = -0.5 * sz + dpml + padding_below + 0.5 * z_thickness

geometry = [mp.Block(
    size=mp.Vector3(xrange[1] - xrange[0], yrange[1] - yrange[0], z_thickness),
    center=mp.Vector3(x_center, y_center, z_center),
    material=material_grid
)]

# --- Light source ---
src_x = x_center - 0.4 * (xrange[1] - xrange[0])
monitor_x = x_center + 0.4 * (xrange[1] - xrange[0])

# --- Sweep settings ---
wavelengths = np.linspace(1.5, 1.6, 10)
frequencies = 1 / wavelengths
trans_flux_values = []
input_flux_values = []

for freq in frequencies:
    sources = [mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=freq),
        center=mp.Vector3(src_x, 0, z_center),
        size=mp.Vector3(0, sy - 2 * dpml, z_thickness),
        direction=mp.X,
        eig_band=1,
        eig_parity=mp.NO_PARITY,
        eig_match_freq=True
    )]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        default_material=sio2
    )

    trans_region = mp.FluxRegion(
        center=mp.Vector3(monitor_x, 0, z_center),
        size=mp.Vector3(0, sy - 2 * dpml, z_thickness),
        direction=mp.X
    )
    trans = sim.add_flux(freq, 0, 1, trans_region)

    input_region = mp.FluxRegion(
        center=mp.Vector3(src_x + 0.05, 0, z_center),
        size=mp.Vector3(0, sy - 2 * dpml, z_thickness),
        direction=mp.X
    )
    refl = sim.add_flux(freq, 0, 1, input_region)

    sim.run(until=500)

    trans_flux = mp.get_fluxes(trans)[0]
    input_flux = mp.get_fluxes(refl)[0]
    trans_flux_values.append(trans_flux)
    input_flux_values.append(input_flux)

    if input_flux != 0:
        print(f"λ = {1/freq:.3f} μm → Transmission = {trans_flux/input_flux:.4f}")
    else:
        print(f"λ = {1/freq:.3f} μm → Input flux = 0")

# --- Plot transmission ---
transmission_ratio = np.array(trans_flux_values) / np.array(input_flux_values)

plt.figure(figsize=(8, 5))
plt.plot(wavelengths, transmission_ratio, marker='o')
plt.xlabel("Wavelength (μm)")
plt.ylabel("Transmission")
plt.title("Transmission Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Visualization: ε and Ey ---
eps_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
Ey_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ey)

z_center_index = eps_data.shape[2] // 2
eps_xy = eps_data[:, :, z_center_index]
Ey_xy = Ey_data[:, :, z_center_index]
x = np.linspace(-0.5 * sx, 0.5 * sx, eps_data.shape[0])
y = np.linspace(-0.5 * sy, 0.5 * sy, eps_data.shape[1])

plt.figure(figsize=(10, 8))
plt.imshow(eps_xy.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
plt.colorbar(label='Dielectric Constant (ε)')
plt.title('Dielectric ε in xy-plane (z=center)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(Ey_xy.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='RdBu')
plt.colorbar(label='Ey Field')
plt.title('Ey Field in xy-plane (z=center)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.tight_layout()
plt.show()
