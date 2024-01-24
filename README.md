# CREASE-2D
Here we list all the codes that were used in the development of CREASE-2D.

sampling_structural_features.py: This generates a list of structural features as described in the main manuscript, used to create the dataset. These values were used as input to generate structures using CASGAP method.

scattering_calc.m: This code calculates the 2D scattering profiles for the structures created using CASGAP method.

crease2D_GA_script.py: This code is the execution of the GA in the CREASE-2D method, which uses an input scattering profile and optimizes the structural features that best match the input scattering profile. It uses the trained ML model.
