# Example 1: Straight Waveguide Simulation using Meep

This example demonstrates how to simulate a straight silicon waveguide using Meep and a GDSII layout.

This simulation script places a continuous wave (CW) source on one of two silicon waveguides drawn in the GDS file, and performs a wave propagation simulation.  
The GDS file is imported as a bitmap, while the waveguide height is defined within the script.


## 📂 Files
- `examples/meep_calculation1.py`: A full 3D simulation of a bent silicon waveguide defined from a GDS file, with MaterialGrid.
- `Bend_waveguide2.gds`: Corresponding layout file used in the example.

- `examples/meep_calculation3.py`:A full 3D simulation of a straight silicon waveguide defined from a GDS file, with MaterialGrid. This script places flux monitors at the input and output sides of the 　　waveguide, and computes the transmission spectrum over the wavelength range from 1.5 to 1.6 µm.
- `straight_waveguide2.gds`: Corresponding layout file used in the example.

## ▶️ How to Run

Install required Python packages:

```bash
pip install meep gdstk shapely matplotlib

python meep_calculation1.py

