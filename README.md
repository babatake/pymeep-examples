# pymeep-examples
pyMeep simulation examples for photonic circuits from GDSII layout files

# Meep Photonic Simulation Examples

This repository contains example simulations using [Meep](https://github.com/probstj/pymeep) — a free FDTD simulation software for modeling electromagnetic systems.

## 📁 Contents

- `examples/meep_calculation1.py`: A full 3D simulation of a bent silicon waveguide defined from a GDS file, with MaterialGrid.
- `Bend_waveguide2.gds`: Corresponding layout file used in the example.

- `examples/meep_calculation3.py`: examples/meep_calculation3.py: A full 3D simulation of a straight silicon waveguide defined from a GDS file, with MaterialGrid. This script places flux monitors at the input and output sides of the 　　waveguide, and computes the transmission spectrum over the wavelength range from 1.5 to 1.6 µm.
- `straight_waveguide2.gds`: Corresponding layout file used in the example.
## 🔧 Requirements

- Python 3.8+
- `meep` (with Python bindings)
- `gdstk`, `shapely`, `numpy`, `matplotlib`

Install dependencies with:

```bash
pip install meep gdstk shapely numpy matplotlib
