# scut_ProFP
A protein fitness predictive framework based on feature combination and intelligent searching


- [scut\_ProFP](#scut_profp)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Acknowledgements](#acknowledgements)
  - [Modifications](#modifications)
  - [Issues](#issues)
  - [Contact](#contact)

## Overview
scut_ProFP is a machine learning predictive framework for protein fitness, designed to guide protein engineering by predicting protein fitness from sequences. It combines feature combination and feature selection techniques: the former to capture rich sequence feature information and the latter to intelligently search for important features. This integration significantly enhances the predictive performance of machine learning models.
![scut_ProFP](doc/overview.png)
## Project Structure
The project directory includes the following key files and folders:  
* data/: Directory containing output results.  
* Datasets/: Directory containing datasets.  
* pySAR/: pySAR is a python library for analyzing sequence activity relationships (SARs)/Sequence function relationships (SFRs) of protein sequences.   
* data.json: json file containing data configuration.  
* Feature_comb.py: Python script for combining features.  
* Shapvalue.py: Python script for calculating SHAP. values.
* Shap_SFS.py: Python script for SHAP-based sequential feature selection.  
## Installation
To set up the project locally, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/scut_ProFP.git
cd scut_ProFP
```
**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

## Usage
**1. Feature Combination.**  
To run the feature combination script, use the following command:
```
python Feature_comb.py
```
**2. Calculate SHAP Values.**  
To calculate SHAP values for your dataset, use the following command:
```bash
python Shapvalue.py
```
**3. SHAP-based Sequential Feature Selection.**  
To perform SHAP-based sequential feature selection, use the following command:
```bash
python Shap_SFS.py
```

## Acknowledgements
The pySAR directory contains source code from the pySAR framework, a Python library for analyzing Sequence Activity Relationships (SARs)/Sequence Function Relationships (SFRs) of protein sequences. We acknowledge the authors of the original work:  
* Research article: "Machine Learning Based Predictive Model for the Analysis of Sequence Activity Relationships Using Protein Spectra and Protein Descriptors" by Mckenna, A., & Dubey, S. (2022).
* pySAR repository:https://github.com/amckenna41/pySAR.
  
For more information, refer to the original research article and pySAR documentation.
## Modifications  
We have made the following modifications to the original pySAR code:  
* descriptor.py: Updated to include additional functionalities and improvements specific to our project's requirements.  
* pySAR.py: Updated to include additional functionalities and improvements specific to our project's requirements.
## Issues
Any issues, errors, or bugs can be raised via the Issues tab in the repository.

## Contact
If you have any questions or comments, please contact us or raise an issue on the Issues tab.