# Multi-Variate Time-Series Transformer

This repo contains the code  required to train the multivariate time-series Transformer.

## Download the data
The Non-Homogeneous Compound Poisson Process aeroelastic simulation data can be downloaded from: [https://doi.org/10.5281/zenodo.5544042](https://doi.org/10.5281/zenodo.5544042)

## Usage
First modify the yaml config file as desired, then the training may be launched with:

`python3 train.py -c path/to/config.yaml`

## Cite
Duthé, G., Abdallah, I., Barber, S., & Chatzi, E. (2021). Modeling and Monitoring Erosion of the Leading Edge of Wind Turbine Blades. Energies, 14(21), 7262.
[https://doi.org/10.3390/en14217262](https://doi.org/10.3390/en14217262)

## Requirements 
- einops>=0.3.0
- h5py>=2.10.0
- numpy>=1.20.1
- python_box>=5.3.0
- PyYAML>=6.0
- torch>=1.8.0
- tqdm>=4.60.0