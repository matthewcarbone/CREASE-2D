# CREASE-2D
Here we list all the codes that were used in the development of CREASE-2D.

__sampling_structural_features.py__: This generates a list of structural features as described in the main manuscript, used to create the dataset. These values were used as input to generate structures using CASGAP method.

__scattering_calc.m__: This code calculates the 2D scattering profiles for the structures created using CASGAP method.

__crease2D_GA_script.py__: This code is the execution of the GA in the CREASE-2D method, which uses an input scattering profile and optimizes the structural features that best match the input scattering profile. It uses the trained ML model.
