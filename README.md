# Computational Reverse Engineering Analysis of Scattering Experiments Method for Interpretation of 2D Small-Angle Scattering Profiles (CREASE-2D)

## Brief Description:
CREASE-2D is the extension of the [CREASE](https://github.com/arthijayaraman-lab/crease_ga) Method which stands for Computational Reverse Engineering Analysis for Scattering Experiments, and is used to interpret detailed structural information, typically of amorphous soft materials, from complete 2D scattering profiles, that are outputed as measurements of small-angle scattering experiments. Isotropic structures in soft materials are interpreted from azimuthally averaged 1D scattering profiles; to understand anisotropic spatial arrangements, however, one has to interpret the entire 2D SAS profile, I(q,θ). The CREASE-2D method interprets I(q,θ) as is, without any averaging about angles, and outputs the relevant structural features using genetic algorithm optimization and an XG-Boost based surrogate ML model for computed scattering profile calculation.

All the key details about this work can be found in the pre-print: [(https://arxiv.org/abs/2401.12381)](https://arxiv.org/abs/2401.12381)
The manuscript is currently under review.

![CREASE-2D](https://github.com/arthijayaraman-lab/CREASE-2D/blob/main/TOC.png)

Here we list all the codes that were used in the development of CREASE-2D.

__sampling_structural_features.py__: This generates a list of structural features as described in the main manuscript, used to create the dataset. These values were used as input to generate 3D structures using CASGAP method.

__Features_6.csv__: This csv file contains the six structural features, i.e., R<sub>μ</sub>, R<sub>σ</sub>, γ<sub>μ</sub>, γ<sub>σ</sub>, κ, φ The first five features are given as input to [CASGAP](https://github.com/arthijayaraman-lab/casgap) to generate 3D structure and φ is calculated after generating the structure.

__scattering_calc.m__: This code calculates the 2D scattering profiles for the structures created using CASGAP method.

__Data_Processing.ipynb__: This Notebook contains the processing of scattering profiles and structural features dividing into train, grid sampling of scattering profiles, making a tabular dataset to train our XGBoost model. For more details please read our [manuscript](https://arxiv.org/abs/2401.12381).

__XGBoost_training.py__: This .py file contains hyperparameter tuning using bayesian optimization and training the XGBoost model. The trained model is given in [here](https://github.com/arthijayaraman-lab/CREASE-2D/blob/main/XGBoost%20Model.zip)

__crease2D_GA_script.py__: This code is the execution of the GA in the CREASE-2D method, which uses an input scattering profile and optimizes the structural features that best match the input scattering profile. It uses the trained ML model.
