# Brain-optimized neural networks learn non-hierarchical models of representation in human visual cortex

[![arXiv shield](https://img.shields.io/badge/bioRxiv-2022.01.21.477293v1-red)](https://www.biorxiv.org/content/10.1101/2022.01.21.477293v1)
[![DOI](https://img.shields.io/badge/DOI-2022.01.21.477293-blue)](https://doi.org/10.1101/2022.01.21.477293)

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)

# Overview

The main purpose of this collection of notebooks is to show how the data-driven neural network model was trained on multiple subject responses, how the transfer learning experiment is performed, how the tuning analysis of the network representation is performed, and how prediction accuracy compares between the different forms of training.

# Repo contents

| File | Description |
|------|-------------|
|`data_preparation.ipynb`| Some preprocessing of the NSD images and responses for convenience. |
|`mask_preparation.ipynb`| Various preprocessing of masks over NSD voxels (not used directly here).|
|`gnet8j_encoding_multisubjects.ipynb`| Training of the joint (all ROI of early visual cortex) data-driven GNet encoding model for all subjects. Includes cross-validation.|
|`gnet8r_encoding_multisubjects.ipynb`| Training of the ROI-wise data-driven GNet encoding model for all subjects. Includes cross-validation.|
|`gnet8x_prediction_multisubjects.ipynb`| Calculation of the model predictions for all images in the NSD dataset.|
|`gnet8j_v_gnet8r_and_transfer_matrix.ipynb`| Validation accuracy comparison between GNet8j and GNet8r and calculation of the experimental transfer learning matrix.|
|`gnet8x_tuning_analysis_and_accuracy.ipynb`| Layerwise and bipartite feature tuning analyses for the GNet8j and GNet8r models and detailed accuracy comparison.|
|`synthetic_transfer_learning.ipynb`| Training of new encoding model (GNet8j, Gnet8r and transfer learning GNet8r) on the prediction of a previous GNet8r trained on NSD. |

# System requirements
## Hardware Requirements
- 128 Gb RAM
- GPU with 12Gb VRAM
Requirements may be lowered for minimal testing enviroments by changing certain routine parameters.

## Software Requirements

The exact minimal requirement are not known exactly. However, the implementation uses standard routines and very few dependencies, therefore old version may still perform adequately. The numerical experiments and analysis has been performed with the following software versions:

- python 3.6.8
- numpy 1.19.5
- scipy 1.5.4
- torch 1.10 with CUDA 11.3 and cudnn 8.2

