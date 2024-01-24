# Title: Computational Reverse Engineering Analysis of Scattering Experiments Method for Interpretation of 2D Small-Angle Scattering Profiles (CREASE-2D)

## Project Description:
The goal of this project is to find the relation between scattering profile of a structure to it's structural properties. For that we first solved of getting scattering profile from given structural features using machine learning and using genetic algorithms we found the possible structural features for given scattering profile using machine learning model as scattering profile generator in it. We used SSIM as our loss function in GA. The overall workflow is shown in TOC figure ![CREASE-2D](https://github.com/arthijayaraman-lab/CREASE-2D/blob/main/TOC.png)

Here we list all the codes that were used in the development of CREASE-2D.

__sampling_structural_features.py__: This generates a list of structural features as described in the main manuscript, used to create the dataset. These values were used as input to generate 3D structures using CASGAP method.

__Features_6.csv__: This csv file contains the six structural features, i.e., R<sub>μ</sub>, R<sub>σ</sub>, γ<sub>μ</sub>, γ<sub>σ</sub>, κ, φ The first five features are given as input to [CASGAP](https://github.com/arthijayaraman-lab/casgap) to generate 3D structure and φ is calculated after generating the structure.

__scattering_calc.m__: This code calculates the 2D scattering profiles for the structures created using CASGAP method.

__Data_Processing.ipynb__: This Notebook contains the processing of scattering profiles and structural features dividing into train, grid sampling of scattering profiles, making a tabular dataset to train our XGBoost model. For more details please read our [manuscript](https://arxiv.org/abs/2401.12381).

__XGBoost_training.py__: This .py file contains hyperparameter tuning using bayesian optimization and training the XGBoost model. The trained model is given in [here](https://github.com/arthijayaraman-lab/CREASE-2D/blob/main/XGBoost%20Model.zip)

__crease2D_GA_script.py__: This code is the execution of the GA in the CREASE-2D method, which uses an input scattering profile and optimizes the structural features that best match the input scattering profile. It uses the trained ML model.
