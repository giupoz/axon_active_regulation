# axon_active_regulation
In this repository, you can find the source code used to simulate in-silico the mathematical modelling of axonal cortex contractility.
The model and the results of the numerical simulations are contained in this paper [[1]](#1).

## Dependencies
The code is developed in Python 3 and tested with version 3.8.5. The following additional libraries are required, in the parentheses we indicate the version used in the simulations reported in the paper:

    FEniCS (version 2019.1.0)
    Numpy (version 1.19.1)
    Scipy (version 1.7.3)

## Repository structure
The repository is structured as follows:

* `axon_class.py`: contains the class `AxonProblem` implementing the nonlinear problem of the axonal cortex contractility described in the paper. In particular the following methods are implemented: 
	* `monitor`: implements the commands to export the outputs of the simulations;
	* `material_properties`: the material properties of the axon (i.e. the shear and the bulk modulus) are defined in this method;
	* `kinematics_features`: the kinematic properties of the axon (i.e. the elastic and inelastic deformation gradients and the Mandel stress tensor) are defined in this method;
	* `strainenergy`: the strain energy functional is implemented in this method;
	* `time_increment_active_strain`: contains the implementation of the time-advancement scheme for the active stretch coefficients; 
	* `find_equilibrium`: returns the stationary solution of the mathematical problem without therapies and stretch. Such solution constitutes the initial configuration adopted to reproduce in-silico the different scenarios;
	* `apply_noco`: implements the therapeutic scenario of nocodazole treatment;
	* `apply_axial_stretch`: implements the mechanical stretch of the axon in the axial direction;
	* `apply_cytoD`: implements the therapeutic scenario of cytochalasin D treatment;

* `scenarios.py`: contains the settings options for all the different mechanical and therapeutic scenarios mentioned in the paper.

To run the simulations it is necessary to set the parameter of the model by choosing the desired scenario from the file `scenarios.py` and run the same file.

## Citing

If you find this code useful for your work, please cite [[1]](#1).

## Licence

The source code contained in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

## References

<a id="1">[1]</a>
D. Andrini, V. Balbi, G. Bevilacqua, G. Lucci, G. Pozzi, D. Riccobelli.
Mathematical modelling of axonal cortex contractility
Brain Multiphysics, 3, 100060 (2022).
