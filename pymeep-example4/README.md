🌀 Nonlinear Waveguide SHG Simulation with Meep (3D FDTD)

This repository contains a 3D FDTD simulation script using Meep to model second-harmonic generation (SHG) in a nonlinear waveguide structure based on χ(2) nonlinearity.
📋 Overview

This script simulates a single-mode waveguide made of a nonlinear χ² medium. A Gaussian mode source excites the waveguide at a fundamental frequency. The simulation computes:

    Transmission at the fundamental frequency

    SHG (second-harmonic) output

    Field distribution of the electric field component Ez

The results help assess SHG conversion efficiency in simple nonlinear photonic waveguides.
🧱 Simulation Structure

    Nonlinear waveguide (block) with χ(2) medium

    Source: EigenModeSource injecting a Gaussian-profiled fundamental mode

    Monitors:

        Input flux monitor near the source

        Transmission monitor at the output

        SHG monitor at 2× the fundamental frequency

    3D simulation including full vectorial fields and PML boundaries

🧪 Key Parameters
Parameter	Value
Refractive index n	3.4
Waveguide width w	0.5 μm
Waveguide height h	0.22 μm
χ² value	1.0 (normalized)
Center frequency	0.64 1/μm (λ ≈ 1.56 μm)
Bandwidth	0.1 1/μm
Number of points	100
Simulation resolution	20 pixels/μm
Simulation domain	3D
📈 Outputs

    Transmission spectrum at the fundamental frequency

    SHG efficiency (power at 2×fcen / power at fcen)

    Ez field distribution in the xy-plane at z = 0

Example plot:

    Blue: Transmission at fundamental frequency

    Orange: SHG conversion efficiency
    (Wavelength scale inverted for optical convention)

▶️ How to Run

Install Meep using conda (recommended):

conda install -c conda-forge meep

Install other dependencies:

pip install matplotlib numpy

Then run:

python nonlinear_shg_simulation.py

🧮 Output Example

After running, the script prints SHG efficiency at λ = 1.55 μm:

SHG efficiency at λ = 1.5500 μm: 1.897222e-13

📝 Notes

    The χ² value here is a relative placeholder and not tied to a specific material system. You can calibrate it using real nonlinear coefficients.

    This simulation is a conceptual starting point and can be extended to include:

        Phase-matching techniques

        Dispersive materials

        Pulsed excitation

        Periodic poling for quasi-phase matching

