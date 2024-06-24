# Structure Tensor Validation: Functions for validating the accuracy of structure tensor analysis for estimating orientations in 2D and 3D images.

## Features

-    Generate 2D and 3D phantoms simulating axons in brain microscopy with parallel and crossing patterns.
-    Perform structure tensor analysis to obtain orientations in 2D and 3D images.
-    Visualize image orientations and their distributions.
-    Estimate k-means of image orientations.
-    Test the accuracy of estimated orientations.

## Installation and Setup
First create and activate a new virtual environment with the following command using venv:
```
python3 -m venv env
source env/bin/activate
```

Or with conda:
```
conda create -n env
conda activate env
```
Note: This code has been tested with python version 3.9. It may not be compatable with other versions.

Then install package requirements:
```
pip install -r requirements.txt
```

For the implementation of k-means with periodic boundary conditions, our package incorporates a module from an unaffiliated Github repository. This must be cloned using the following command after nagivating to the structure_tensor_validation root directory:
```
git clone https://github.com/kpodlaski/periodic-kmeans.git
```
The details of their method are published in the article: Miniak-Górecka, A.; Podlaski, K.; Gwizdałła, T. Using K-Means Clustering in Python with Periodic Boundary Conditions. Symmetry 2022, 14, 1237. https://doi.org/10.3390/sym14061237

## Usage
Open the Jupyter notebook run_st_analysis_validation.ipynb for an example walkthrough of our validation pipeline.

## Supplementary Data
The unaltered output data from structure tensor analysis validation is located in the outputs folder. These are stored in comma separated files where each row corresponds to a different parameter configuration. The phantom and structure tensor analysis parameters are listed in the first columns and the resulting error for each setting is in the last column.

The example_microscopy_data folder contains the two example microscopy patches &mdash one 2D and one 3D &mdash that were used to illustrate the relationship between simulated phantoms and real image data.
