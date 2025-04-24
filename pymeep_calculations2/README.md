🧠 Adjoint Optimization of a 3D-waveguide-based directional coupler Structure Using Meep and B-Splines

This repository demonstrates an adjoint-based inverse design framework for optimizing the geometry of a 2D photonic waveguide using Meep, the open-source FDTD simulator. The device geometry is parameterized by B-spline curves, and the forward and adjoint electromagnetic fields are used to compute shape gradients with respect to the structure.
✨ Features

    Smooth boundary modeling using cubic B-splines

    MaterialGrid-based dielectric interpolation

    Adjoint method for efficient gradient computation

    Fully implemented in Python using Meep and NumPy

📐 Problem Setup

We consider a 2D waveguide structure formed by upper and lower B-spline curves:

    The top and bottom boundaries are controlled by a set of 12 control points.

    Only a subset (indices 3 to 8) of these control points are allowed to be optimized.

    The waveguide is embedded in SiO₂ (n=1.44), while the core material is Si (n=3.48).

The B-spline boundary shapes are evaluated and rasterized into a MaterialGrid, which is passed to Meep for simulation.
🖼️ B-spline Example Shape

Here's a plot of the initial top and bottom spline boundaries:

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Parameters
num_ctrl = 12
length = 30.0
x_ctrl = np.linspace(0, length, num_ctrl)
base_y_top = 1.5
base_y_bot = -1.5
margin = 0.1
params_top = np.zeros(num_ctrl)
params_bot = np.zeros(num_ctrl)

# Generate splines
full_top = base_y_top + margin + params_top
full_bot = base_y_bot - margin - params_bot
knot_vector = np.linspace(0, 1, num_ctrl + 3 + 1)
spline_top = BSpline(knot_vector, full_top, 3)
spline_bot = BSpline(knot_vector, full_bot, 3)

t_vals = np.linspace(0, 1, 200)
x_vals = np.linspace(0, length, 200)
y_vals_top = spline_top(t_vals)
y_vals_bot = spline_bot(t_vals)

# Plot
plt.figure(figsize=(10, 3))
plt.plot(x_vals, y_vals_top, label="Top Boundary", linewidth=2)
plt.plot(x_vals, y_vals_bot, label="Bottom Boundary", linewidth=2)
plt.fill_between(x_vals, y_vals_bot, y_vals_top, color='gray', alpha=0.3)
plt.title("Initial Waveguide Geometry Using B-splines")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

⚙️ Optimization Loop

The optimization is performed using the adjoint method:

    Forward simulation with a mode source → get Ey_fwd.

    Adjoint simulation with source at output → get Ey_adj.

    Compute gradient:
    ∂J∂ϵ=2 Re(Eyfwd⋅Eyadj)
    ∂ϵ∂J​=2Re(Eyfwd​⋅Eyadj​)

    Integrate the gradient over regions affected by each control point.

    Update the design variables via gradient descent.

📁 File Overview
File	Description
adjoint_opt.py	Main optimization script using Meep and B-splines
README.md (this)	Explanation of the optimization process
requirements.txt	Python dependencies (Meep, NumPy, etc.)
🔧 Requirements

    Python ≥ 3.8

    meep (GPU or CPU version)

    numpy, matplotlib, scipy, shapely

Install with:

pip install -r requirements.txt

🧪 Example Output

After several iterations, the shape adapts to maximize the output flux (transmission). Here's a sample log:

=== Iteration 1/5 ===
Forward J = 0.084235
∂J/∂x_top: [...]
∂J/∂x_bot: [...]
Updated design_params_top: [...]

🔬 Applications

This type of adjoint-based design is widely used in:

    Photonic inverse design

    Topology optimization

    Silicon photonics layout automation

📮 Contact

If you use or build upon this code, feel free to open an issue or reach out. Collaboration is welcome!
