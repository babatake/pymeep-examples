# Example 1: Straight Waveguide Simulation using Meep

This example demonstrates how to simulate a straight silicon waveguide using Meep and a GDSII layout.

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
