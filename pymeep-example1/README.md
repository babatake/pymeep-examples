# Example 1: Straight Waveguide Simulation using Meep

This example demonstrates how to simulate a straight silicon waveguide using Meep and a GDSII layout.

This simulation script places a continuous wave (CW) source on one of two silicon waveguides drawn in the GDS file, and performs a wave propagation simulation.  
The GDS file is imported as a bitmap, while the waveguide height is defined within the script.


## 📂 Files

| File | Description |
|------|-------------|
| `meep_calculation1.py` | Python script that sets up the simulation domain, materials, and runs the Meep FDTD simulation. |
| `straight_waveguide2.gds` | GDSII layout file describing the straight waveguide geometry. |

## ▶️ How to Run

Install required Python packages:

```bash
pip install meep gdstk shapely matplotlib

python meep_calculation1.py

