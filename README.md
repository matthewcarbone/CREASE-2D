# Computational Reverse Engineering Analysis of Scattering Experiments Method for Interpretation of 2D Small-Angle Scattering Profiles (CREASE-2D)

## Brief Description:
CREASE-2D is the extension of the [CREASE](https://github.com/arthijayaraman-lab/crease_ga) Method which stands for Computational Reverse Engineering Analysis for Scattering Experiments, and is used to interpret detailed structural information, typically of amorphous soft materials, from complete 2D scattering profiles, that are outputed as measurements of small-angle scattering experiments. Isotropic structures in soft materials are interpreted from azimuthally averaged 1D scattering profiles; to understand anisotropic spatial arrangements, however, one has to interpret the entire 2D SAS profile, I(q,θ). The CREASE-2D method interprets I(q,θ) as is, without any averaging about angles, and outputs the relevant structural features using genetic algorithm optimization and an XG-Boost based surrogate ML model for computed scattering profile calculation.

All the key details about this work can be found in the pre-print: [(https://arxiv.org/abs/2401.12381)](https://arxiv.org/abs/2401.12381)
The manuscript is currently under review.

![CREASE-2D](https://github.com/arthijayaraman-lab/CREASE-2D/blob/main/TOC.png)

Here we list all the codes that were used in the development of CREASE-2D.

__sampling_structural_features.py__: This python script generates a list of structural features as described in the main manuscript, and is used to create the dataset for training and validation of the ML model. The structural featuresare directly used as inputs to generate 3D structures using the [CASGAP](https://github.com/arthijayaraman-lab/casgap) method. The CSV file __Features_6.csv__ contains a tabulated list of 6 structural features:  R<sub>μ</sub>, R<sub>σ</sub>, γ<sub>μ</sub>, γ<sub>σ</sub>, κ and φ, that were used in the dataset shared in the main manuscript.

__scattering_calc.m__: This MATLAB script contains the parallelized code which calculates the 2D scattering profiles for the structures generated using the CASGAP method.

__Data_Processing.ipynb__: This Jupyter Notebook contains steps used to process the dataset of scattering profiles and all structural features, to split into training and testing, and to subsample the data, as described in the manuscript, which is then used as input for training the XGBoost model.

__XGBoost_training.py__: This python script contain the implementation of Bayesian optimization search using hyperparameter tuning and training the XGBoost model. The trained model is provided as a zip file _XGBoost 20Model.zip_.

__crease2D_GA_script.py__: This python script describes the coded implementation of the genetic algorithm used to execute the CREASE-2D method. The code uses as input an 'experimnetal' scattering profile and optimizes the structural features that when provided to the trained ML model, provide 'computed' scattering profiles which closely resemble the input 'experimental' scattering profile.
