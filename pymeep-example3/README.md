Ring Resonator Simulation with Meep (3D FDTD)

This repository contains a 3D FDTD simulation script using Meep to analyze the spectral response of a silicon ring resonator coupled to bus and drop waveguides.
📌 Features

    3D simulation of a ring resonator with vertical confinement

    Coupled bus and drop waveguides

    Gaussian mode excitation using EigenModeSource

    Transmission and drop port flux spectrum

    Ez field visualization in the xy-plane (z = 0)

🌀 Structure

The simulation geometry includes:

    A ring resonator of radius r = 10 μm and waveguide width w = 0.5 μm

    A bus waveguide and a drop waveguide (both with the same width and refractive index)

    A gap of 0.05 μm between the ring and waveguides

    A 3D cell with PML boundaries and vertical thickness to account for realistic confinement

🧪 Simulation Parameters
Parameter	Value
Refractive index (n)	3.4
Waveguide width (w)	0.5 μm
Height (h)	0.22 μm
Ring radius (r)	10.0 μm
Gap	0.05 μm
PML thickness	2.0 μm
Frequency center	0.64 1/μm
Frequency width	0.1 1/μm
Number of frequencies	50
Resolution	40 pixels/μm
📊 Output

    Spectral Response: The script plots the transmission and drop port spectra as functions of wavelength.

    Field Distribution: Ez field distribution at z = 0 plane is visualized using matplotlib.

▶️ Requirements

    Python 3.8+

    Meep

    numpy, matplotlib

Install Meep via conda (recommended):

conda install -c conda-forge meep

🚀 How to Run

python ring_resonator_meep.py

Make sure you are running in an environment where Meep is correctly installed (e.g., Conda environment).
📝 Notes

    This example is useful for basic ring resonator studies and can be extended to include material dispersion, gain, or thermal effects.

    For better performance, consider switching to 2.5D simulations if vertical confinement is not critical.
